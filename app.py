from flask import Flask
from flask_socketio import SocketIO
import threading
import time
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

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

def send_emotion_message():
    """Send emotion message to all connected clients"""
    global latest_emotion, latest_confidence
    if connected_clients > 0:
        response = emotion_response.get(latest_emotion, "How are you?")
        payload = {
            "type": "AVATAR_TALK",
            "text": response,
            "confidence": round(latest_confidence, 4),
            "emotion": latest_emotion
        }
        print("Sending via socket.io:", payload)
        socketio.emit("AVATAR_TALK", payload)

def camera_worker():
    global latest_emotion, latest_confidence
    cap = cv2.VideoCapture(0)
    last_emotion = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            latest_emotion = emotion_dict[maxindex]
            latest_confidence = float(prediction[0][maxindex])
            print(f"Detected: {latest_emotion} ({latest_confidence:.2f})")
            
            # Send message immediately when emotion changes
            if latest_emotion != last_emotion and connected_clients > 0:
                send_emotion_message()
                last_emotion = latest_emotion
            break
        time.sleep(0.5)

def emotion_sender():
    """Send periodic emotion updates"""
    while True:
        time.sleep(10)
        if connected_clients > 0:
            send_emotion_message()

@app.route("/")
def index():
    return "Emotion detection running with confidence score."

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")
    
    # Send current emotion state immediately to new client
    send_emotion_message()

@socketio.on("disconnect")
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"Client disconnected. Total clients: {connected_clients}")

if __name__ == "__main__":
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=emotion_sender, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=3001)
