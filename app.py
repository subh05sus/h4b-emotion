from flask import Flask
from flask_socketio import SocketIO
import threading
import time
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import logging

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

# Model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load model weights
model.load_weights('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

def get_emotion_response(emotion, confidence):
    """Generate emotion-specific responses based on confidence levels"""
    
    # High confidence responses (80%+)
    high_confidence_responses = {
        "Angry": [
            "I can clearly see you're very angry right now. Take deep breaths and count to ten.",
            "Your anger is quite evident. What's causing this intense frustration?",
            "You look really upset. Would you like to talk about what's making you so angry?"
        ],
        "Disgusted": [
            "You look genuinely disgusted. Is something particularly unpleasant bothering you?",
            "That's a clear look of disgust. What's causing this strong reaction?",
            "I can see you're really put off by something. What's making you feel this way?"
        ],
        "Fearful": [
            "I can see you're quite frightened. Remember, you're safe here with me.",
            "You look genuinely scared. Take a moment to breathe - everything will be okay.",
            "Your fear is very apparent. What's making you feel so anxious right now?"
        ],
        "Happy": [
            "You're beaming with joy! Your happiness is absolutely contagious!",
            "What a wonderful, genuine smile! You look truly delighted!",
            "Your happiness is radiating! Something amazing must have happened!"
        ],
        "Neutral": [
            "You appear completely calm and composed. A picture of tranquility.",
            "Your expression shows perfect balance and serenity.",
            "You look peacefully neutral - very centered and mindful."
        ],
        "Sad": [
            "I can see deep sadness in your expression. I'm here if you want to share what's wrong.",
            "You look genuinely heartbroken. Sometimes it helps to talk about what's troubling you.",
            "Your sadness is very clear. Remember that difficult feelings are temporary."
        ],
        "Surprised": [
            "Wow! You look absolutely astonished! What just happened?",
            "That's pure shock on your face! Tell me about this big surprise!",
            "You're clearly stunned by something! What caught you so off guard?"
        ]
    }
    
    # Medium confidence responses (50-79%)
    medium_confidence_responses = {
        "Angry": [
            "You seem somewhat irritated. Is something bothering you?",
            "I detect some frustration. What's on your mind?",
            "You appear a bit upset. Want to talk about it?"
        ],
        "Disgusted": [
            "You look a bit put off by something. What's not sitting well with you?",
            "I sense some displeasure. Is something not quite right?",
            "You seem mildly disgusted. What's causing this reaction?"
        ],
        "Fearful": [
            "You look a bit worried. Is everything alright?",
            "I sense some anxiety. What's making you feel uneasy?",
            "You appear somewhat concerned. What's troubling you?"
        ],
        "Happy": [
            "You seem to be in a good mood! Something nice happen?",
            "I can see a hint of happiness. What's bringing you joy?",
            "You look pleased about something. Care to share?"
        ],
        "Neutral": [
            "You appear calm and collected. How are you feeling?",
            "Your expression seems balanced. Everything going well?",
            "You look composed. What's on your mind?"
        ],
        "Sad": [
            "You seem a bit down. Is something weighing on your mind?",
            "I detect some melancholy. What's making you feel low?",
            "You appear somewhat troubled. Want to talk about it?"
        ],
        "Surprised": [
            "You look a bit taken aback. Something unexpected happen?",
            "I sense some surprise. What caught your attention?",
            "You seem mildly shocked. What's the news?"
        ]
    }
    
    # Low confidence responses (below 50%)
    low_confidence_responses = {
        "Angry": [
            "I'm picking up on some possible tension. How are you really feeling?",
            "There might be some frustration there. Want to talk about your day?",
            "I sense you might be a bit agitated. What's going on?"
        ],
        "Disgusted": [
            "Something seems to be bothering you slightly. What's on your mind?",
            "I might be detecting some displeasure. Is everything okay?",
            "You could be feeling a bit off about something. Care to share?"
        ],
        "Fearful": [
            "You might be feeling a bit uncertain. Is there something worrying you?",
            "I sense you could be slightly anxious. What's on your mind?",
            "There might be some concern there. Want to talk about it?"
        ],
        "Happy": [
            "You might be feeling a bit positive. What's going well today?",
            "I think I detect some contentment. How are things going?",
            "You could be in a decent mood. Anything good happening?"
        ],
        "Neutral": [
            "You seem fairly balanced. How are you feeling right now?",
            "Your expression appears neutral. What's going through your mind?",
            "You look pretty calm. How's your day going?"
        ],
        "Sad": [
            "You might be feeling a bit low. Is everything alright?",
            "I think I detect some sadness. What's been on your mind?",
            "You could be feeling down about something. Want to share?"
        ],
        "Surprised": [
            "You might have been caught off guard by something. What happened?",
            "I think something unexpected might have occurred. Care to share?",
            "You could be processing something surprising. What's new?"
        ]
    }
    
    # Select response set based on confidence
    if confidence >= 0.8:
        responses = high_confidence_responses.get(emotion, ["I can clearly see strong emotions."])
    elif confidence >= 0.5:
        responses = medium_confidence_responses.get(emotion, ["I think I detect some emotions."])
    else:
        responses = low_confidence_responses.get(emotion, ["I'm trying to read your expression."])
    
    # Add confidence qualifier to the response
    import random
    base_response = random.choice(responses)

    
    return base_response

latest_emotion = "Neutral"
latest_confidence = 0.0
connected_clients = 0
camera_active = False
last_message_time = 0

# Aggressive frequency control
emotion_stability_tracker = {}
last_sent_emotion = None
minimum_emotion_duration = 15  # Must detect same emotion for 15 consecutive times (7.5 seconds)
message_cooldown = 30  # 30 seconds between any messages
message_count = 0

def should_send_message(emotion, confidence):
    """Determine if we should send a message - very strict criteria"""
    global emotion_stability_tracker, last_sent_emotion, minimum_emotion_duration
    
    # Track emotion stability
    if emotion not in emotion_stability_tracker:
        emotion_stability_tracker[emotion] = 0
    
    # Reset other emotions when new emotion detected
    for key in list(emotion_stability_tracker.keys()):
        if key != emotion:
            emotion_stability_tracker[key] = 0
    
    # Increment current emotion count
    emotion_stability_tracker[emotion] += 1
    
    current_count = emotion_stability_tracker[emotion]
    
    # Only send message if:
    # 1. Emotion has been stable for minimum duration
    # 2. Haven't sent this emotion recently
    # 3. Confidence is reasonably high (> 0.6)
    # 4. Enough time has passed since last message
    if (current_count >= minimum_emotion_duration and 
        emotion != last_sent_emotion and 
        confidence > 0.6 and
        (time.time() - last_message_time) >= message_cooldown):
        
        print(f"Message criteria met: {emotion} stable for {current_count} detections, confidence {confidence:.2f}")
        return True
    
    return False

def send_emotion_message():
    """Send emotion message to all connected clients with error handling"""
    global latest_emotion, latest_confidence, last_message_time, message_count, last_sent_emotion
    
    current_time = time.time()
    
    if connected_clients > 0:
        try:
            response = get_emotion_response(latest_emotion, latest_confidence)
            message_count += 1
            payload = {
                "type": "AVATAR_TALK",
                "text": response,
                "confidence": round(latest_confidence, 4),
                "emotion": latest_emotion,
                "confidence_level": "high" if latest_confidence >= 0.8 else "medium" if latest_confidence >= 0.5 else "low",
                "message_number": message_count,
                "stability_count": emotion_stability_tracker.get(latest_emotion, 0),
                "timestamp": int(current_time * 1000)
            }
            print(f"🔊 Sending message #{message_count}: {latest_emotion} (confidence: {latest_confidence:.2f})")
            socketio.emit("AVATAR_TALK", payload)
            last_message_time = current_time
            last_sent_emotion = latest_emotion
        except Exception as e:
            print(f"Error sending message: {e}")

def camera_worker():
    global latest_emotion, latest_confidence, camera_active
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                      # Suppress prediction warnings
                    with np.errstate(divide='ignore', invalid='ignore'):
                        prediction = model.predict(cropped_img, verbose=0)
                    
                    maxindex = int(np.argmax(prediction))
                    latest_emotion = emotion_dict[maxindex]
                    latest_confidence = float(prediction[0][maxindex])
                    
                    # Show detection with stability info
                    stability_count = emotion_stability_tracker.get(latest_emotion, 0) + 1
                    print(f"👁️  Detected: {latest_emotion} ({latest_confidence:.2f}) [Stability: {stability_count}/{minimum_emotion_duration}]")
                    
                    # Only send message if strict criteria are met
                    if should_send_message(latest_emotion, latest_confidence) and connected_clients > 0:
                        send_emotion_message()
                    break
                    
                time.sleep(0.5)  # Reduce CPU usage
                
            except Exception as e:
                print(f"Error in camera processing: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"Camera initialization error: {e}")
    finally:
        if cap:
            cap.release()
        camera_active = False

def emotion_sender():
    """Send very infrequent heartbeat - no emotion messages"""
    while True:
        time.sleep(120)  # Only heartbeat every 2 minutes
        if connected_clients > 0:
            try:
                # Send heartbeat/status only - NO emotion messages
                socketio.emit("heartbeat", {
                    "status": "alive", 
                    "clients": connected_clients,
                    "total_messages": message_count,
                    "last_emotion": latest_emotion,
                    "last_confidence": round(latest_confidence, 2)
                })
                print(f"💓 Heartbeat sent - {connected_clients} clients connected, {message_count} total messages")
                
            except Exception as e:
                print(f"Error in heartbeat: {e}")

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
        "total_messages_sent": message_count,
        "last_sent_emotion": last_sent_emotion,
        "emotion_stability": emotion_stability_tracker,
        "message_cooldown_seconds": message_cooldown,
        "minimum_stability_required": minimum_emotion_duration
    }

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")
    
    try:
        # Send current emotion state immediately to new client
        socketio.emit("connection_ack", {"message": "Connected successfully"})
        # DON'T send emotion message on connect - let them wait for natural detection
        print(f"✅ Client connected - no immediate message sent")
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
        socketio.emit("pong", {"timestamp": int(time.time() * 1000)})
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

import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    print("Starting emotion detection server...")
    
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
