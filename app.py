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

emotion_response = {
    "Angry": "Take a deep breath, what's bothering you?",
    "Disgusted": "Is something unpleasant around you?",
    "Fearful": "Don't be afraid, you're safe here.",
    "Happy": "You look happy!",
    "Neutral": "You seem calm and neutral.",
    "Sad": "Why are you sad?",
    "Surprised": "That face shows surprise! What's the news?"
}

latest_emotion = "Neutral"
latest_confidence = 0.0
connected_clients = 0
camera_active = False
last_message_time = 0

def send_emotion_message():
    """Send emotion message to all connected clients with error handling"""
    global latest_emotion, latest_confidence, last_message_time
    
    current_time = time.time()
    # Avoid sending messages too frequently (max once per second)
    if current_time - last_message_time < 1.0:
        return
        
    if connected_clients > 0:
        try:
            response = emotion_response.get(latest_emotion, "How are you?")
            payload = {
                "type": "AVATAR_TALK",
                "text": response,
                "confidence": round(latest_confidence, 4),
                "emotion": latest_emotion,
                "timestamp": int(current_time * 1000)
            }
            print("Sending via socket.io:", payload)
            socketio.emit("AVATAR_TALK", payload)
            last_message_time = current_time
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
                    print(f"Detected: {latest_emotion} ({latest_confidence:.2f})")
                    
                    # Send message immediately when emotion changes
                    if latest_emotion != last_emotion and connected_clients > 0:
                        send_emotion_message()
                        last_emotion = latest_emotion
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
    """Send periodic emotion updates with heartbeat"""
    while True:
        time.sleep(15)  # Increased interval to reduce load
        if connected_clients > 0:
            try:
                # Send heartbeat/status message
                socketio.emit("heartbeat", {"status": "alive", "clients": connected_clients})
                
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
        "confidence": latest_confidence
    }

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")
    
    try:
        # Send current emotion state immediately to new client
        socketio.emit("connection_ack", {"message": "Connected successfully"})
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
