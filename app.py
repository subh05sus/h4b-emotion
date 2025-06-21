from flask import Flask
from flask_socketio import SocketIO
import threading
import time
import cv2
import numpy as np
import asyncio
import base64
import io
import os
import PIL.Image
import logging
import json
from google import genai
from google.genai import types

# Configure logging
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

app = Flask(__name__)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10000,
    logger=False,
    engineio_logger=False,
    async_mode='threading'
)

# Gemini API configuration
MODEL = "gemini-1.5-flash"
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyCTVlbNUi8NGVyzdxqMsH7kwfofWLHKmFQ"),
)

# Global variables
latest_emotion = "Neutral"
latest_confidence = 0.0
latest_objects = []
latest_message = "Looking around to see what's happening..."
connected_clients = 0
camera_active = False
last_message_time = 0
gemini_session = None

# Context tracking for previous analyses
emotion_history = []
object_history = []
analysis_context = {
    "previous_emotions": [],
    "emotion_trends": "",
    "object_consistency": [],
    "interaction_count": 0,
    "session_start_time": time.time()
}

def analyze_emotion_and_objects_with_gemini(image_data):
    """Analyze emotion and objects using Gemini Vision API"""
    try:
        # Get context from previous analyses
        context_summary = get_context_summary()
        
        # Create the prompt for emotion and object analysis
        prompt = f"""    
        Analyze this image and provide:
        1. The primary emotion of any person in the image
        2. Key objects visible in the scene
        3. Generate a casual, human-like response message based on what you see
        
        CONTEXT FROM PREVIOUS INTERACTIONS: {context_summary}
        
        Use this context to make your response more natural and acknowledge patterns or changes you notice.
        
        Respond ONLY with a JSON object in this exact format:
        {{
            "emotion": "one of: Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral",
            "confidence": 0.85,
            "objects": ["object1", "object2", "object3"],
            "message": "casual human-like message about what you observe - be conversational and natural, consider the context from previous interactions"
        }}
        
        For the message:
        - Use casual, natural language like you're talking to a friend
        - Comment on emotions AND objects you see
        - Consider the context and acknowledge patterns, changes, or consistency
        - Vary your conversation starters
        - Be encouraging and positive when appropriate
        - Ask questions or make observations naturally
        - Keep it conversational and engaging
        - Reference previous emotions or trends when relevant
        
        Examples of good context-aware messages:
        - "Still looking happy! I can see you're keeping that positive energy going."
        - "I notice you've shifted from neutral to more focused - that laptop is getting some serious attention!"
        - "You seem to be in a good mood today - this is the third time I've seen you smiling."
        - "Your workspace looks consistently organized, and you're looking more relaxed now."
        
        Be accurate with confidence scores (0.0 to 1.0). Only use high confidence (0.8+) when very certain about the emotion.
        """
        
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": image_data}
                    ]
                }
            ]
        )
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        try:
            # Look for JSON content between ```json and ``` or just raw JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
                
            result = json.loads(json_text)
            
            # Validate the response structure
            if "emotion" in result and "confidence" in result and "message" in result:
                return (
                    result["emotion"], 
                    float(result["confidence"]), 
                    result.get("objects", []),
                    result["message"]
                )
            else:
                print(f"Invalid response structure: {result}")
                return "Neutral", 0.5, [], "I'm having trouble reading your expression right now."
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response_text}")
            return "Neutral", 0.3, [], "Something went wrong while analyzing the image."
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Neutral", 0.2, [], f"Having some technical difficulties: {str(e)}"

def prepare_image_for_gemini(frame):
    """Convert OpenCV frame to Gemini-compatible format"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        
        # Resize for better processing
        img.thumbnail([1024, 1024])
        
        # Convert to base64
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        
        image_bytes = image_io.read()
        return {
            "mime_type": "image/jpeg", 
            "data": base64.b64encode(image_bytes).decode()
        }
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None

def send_emotion_message():
    """Send emotion message to all connected clients with error handling"""
    global latest_emotion, latest_confidence, latest_message, last_message_time
    
    current_time = time.time()
    # Avoid sending messages too frequently (max once every 30 seconds)
    if current_time - last_message_time < 30.0:
        return
        
    if connected_clients > 0:
        try:
            payload = {
                "type": "AVATAR_TALK",
                "text": latest_message
            }
            print("Sending via socket.io:", payload)
            socketio.emit("AVATAR_TALK", payload)
            last_message_time = current_time
        except Exception as e:
            print(f"Error sending message: {e}")

def camera_worker():
    global latest_emotion, latest_confidence, latest_objects, latest_message, camera_active
    camera_active = True
    cap = None
    last_emotion = None
    consecutive_failures = 0
    
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
        
        while camera_active:
            try:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print("Camera connection lost, retrying...")
                        cap.release()
                        time.sleep(2)
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                    
                consecutive_failures = 0
                
                # Prepare image for Gemini analysis
                image_data = prepare_image_for_gemini(frame)
                if image_data:
                    emotion, confidence, objects, message = analyze_emotion_and_objects_with_gemini(image_data)
                    
                    # Update context tracking
                    update_analysis_context(emotion, confidence, objects)
                    
                    latest_emotion = emotion
                    latest_confidence = confidence
                    latest_objects = objects
                    latest_message = message
                    
                    print(f"Gemini detected: {emotion} ({confidence:.2f}) - Objects: {objects}")
                    print(f"Generated message: {message}")
                    
                    # Send message only when emotion changes significantly or confidence is good
                    if (emotion != last_emotion or confidence > 0.6) and connected_clients > 0:
                        send_emotion_message()
                        last_emotion = emotion
                    
                time.sleep(12.0)  # Increased to 12 seconds for less frequent analysis
                
            except Exception as e:
                print(f"Error in camera processing: {e}")
                time.sleep(2)
                
    except Exception as e:
        print(f"Camera initialization error: {e}")
    finally:
        if cap:
            cap.release()
        camera_active = False

def emotion_sender():
    """Send periodic emotion updates"""
    while True:
        time.sleep(60)  # Set to 60 seconds to match message frequency
        if connected_clients > 0:
            try:
                # Send emotion update if we have valid data
                if latest_confidence > 0:
                    send_emotion_message()
            except Exception as e:
                print(f"Error in emotion sender: {e}")

@app.route("/")
def index():
    return "Emotion detection running with confidence score."

@app.route("/health")
def health():
    return {
        "status": "running",
        "clients": connected_clients,
        "camera_active": camera_active,
        "latest_emotion": latest_emotion,
        "confidence": latest_confidence,
        "latest_objects": latest_objects,
        "latest_message": latest_message,
        "context": {
            "interaction_count": analysis_context["interaction_count"],
            "emotion_trends": analysis_context["emotion_trends"],
            "previous_emotions": analysis_context["previous_emotions"],
            "consistent_objects": analysis_context["object_consistency"],
            "session_duration_minutes": round((time.time() - analysis_context["session_start_time"]) / 60, 1)
        }
    }

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")
    
    try:
        # Send current emotion state immediately to new client
        if latest_confidence > 0:
            send_emotion_message()
    except Exception as e:
        print(f"Error on connect: {e}")

@socketio.on("disconnect")
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"Client disconnected. Total clients: {connected_clients}")

@socketio.on("ping")
def handle_ping():
    """Handle client ping requests"""
    try:
        # Simple pong response with same structure
        socketio.emit("AVATAR_TALK", {
            "type": "AVATAR_TALK",
            "text": "Connection is active"
        })
    except Exception as e:
        print(f"Error handling ping: {e}")

@socketio.on_error_default
def default_error_handler(e):
    print(f"Socket.IO error: {e}")

def cleanup():
    """Cleanup function"""
    global camera_active
    camera_active = False
    print("Cleaning up resources...")

def update_analysis_context(emotion, confidence, objects):
    """Update the context tracking with new analysis data"""
    global emotion_history, object_history, analysis_context
    
    current_time = time.time()
    
    # Add to emotion history (keep last 10 entries)
    emotion_entry = {
        "emotion": emotion,
        "confidence": confidence,
        "timestamp": current_time
    }
    emotion_history.append(emotion_entry)
    if len(emotion_history) > 10:
        emotion_history.pop(0)
    
    # Add to object history (keep last 5 entries)
    object_history.append({
        "objects": objects,
        "timestamp": current_time
    })
    if len(object_history) > 5:
        object_history.pop(0)
    
    # Update analysis context
    analysis_context["previous_emotions"] = [e["emotion"] for e in emotion_history[-5:]]
    analysis_context["interaction_count"] += 1
    
    # Determine emotion trends
    if len(emotion_history) >= 3:
        recent_emotions = [e["emotion"] for e in emotion_history[-3:]]
        if all(e == recent_emotions[0] for e in recent_emotions):
            analysis_context["emotion_trends"] = f"consistently {recent_emotions[0].lower()}"
        elif emotion_history[-1]["emotion"] != emotion_history[-2]["emotion"]:
            analysis_context["emotion_trends"] = f"changed from {emotion_history[-2]['emotion'].lower()} to {emotion.lower()}"
        else:
            analysis_context["emotion_trends"] = "showing mixed emotions"
    
    # Track consistent objects
    if len(object_history) >= 2:
        all_recent_objects = []
        for entry in object_history[-3:]:
            all_recent_objects.extend(entry["objects"])
        # Find objects that appear frequently
        from collections import Counter
        object_counts = Counter(all_recent_objects)
        analysis_context["object_consistency"] = [obj for obj, count in object_counts.items() if count >= 2]

def get_context_summary():
    """Generate a context summary for Gemini"""
    if analysis_context["interaction_count"] == 0:
        return "This is the first interaction."
    
    context_parts = []
    
    # Add interaction count
    context_parts.append(f"This is interaction #{analysis_context['interaction_count']}")
    
    # Add emotion history
    if analysis_context["previous_emotions"]:
        prev_emotions = ", ".join(analysis_context["previous_emotions"][-3:])
        context_parts.append(f"Recent emotions: {prev_emotions}")
    
    # Add emotion trends
    if analysis_context["emotion_trends"]:
        context_parts.append(f"Emotion trend: {analysis_context['emotion_trends']}")
    
    # Add consistent objects
    if analysis_context["object_consistency"]:
        consistent_objects = ", ".join(analysis_context["object_consistency"][:3])
        context_parts.append(f"Consistent objects in scene: {consistent_objects}")
    
    # Add session duration
    session_duration = (time.time() - analysis_context["session_start_time"]) / 60
    if session_duration > 5:
        context_parts.append(f"Session duration: {session_duration:.0f} minutes")
    
    return " | ".join(context_parts)

def reset_analysis_context():
    """Reset the analysis context (useful for new sessions)"""
    global emotion_history, object_history, analysis_context
    
    emotion_history.clear()
    object_history.clear()
    analysis_context = {
        "previous_emotions": [],
        "emotion_trends": "",
        "object_consistency": [],
        "interaction_count": 0,
        "session_start_time": time.time()
    }
    print("Analysis context has been reset")

@app.route("/reset-context", methods=["POST"])
def reset_context_endpoint():
    """Endpoint to reset the analysis context"""
    reset_analysis_context()
    return {"status": "context reset", "message": "Analysis context has been cleared"}

import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    print("Starting Gemini-powered emotion detection server...")
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key:")
        print("On Windows: set GEMINI_API_KEY=your_api_key_here")
        print("On Linux/Mac: export GEMINI_API_KEY=your_api_key_here")
        exit(1)
    
    try:
        # Start camera worker thread
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        camera_thread.start()
        
        # Start emotion sender thread
        sender_thread = threading.Thread(target=emotion_sender, daemon=True)
        sender_thread.start()
        
        # Run the app with optimized settings
        socketio.run(
            app, 
            host="0.0.0.0", 
            port=3001,
            debug=False,
            use_reloader=False,
            log_output=False
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        cleanup()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup()
