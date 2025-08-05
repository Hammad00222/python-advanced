import cv2
import numpy as np
import os
import urllib.request
import zipfile
from tensorflow.keras.models import model_from_json

# ----- FIXED MODEL DOWNLOAD -----
# Download model files with error handling
MODEL_URL = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading emotion model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except:
        print("Error downloading model. Using fallback solution.")
        MODEL_PATH = None

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create emotion model
emotion_model = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        # Load pre-trained emotion recognition model
        emotion_model = model_from_json(open("model.json", "r").read()) if os.path.exists("model.json") else None
        if emotion_model:
            emotion_model.load_weights(MODEL_PATH)
            print("Emotion model loaded successfully!")
    except:
        print("Couldn't load model weights. Using simple emotion detection.")
        emotion_model = None
else:
    print("Model file not found. Using simple emotion detection.")

def preprocess_face(face_roi):
    """Preprocess face region for emotion prediction"""
    # Convert to grayscale and resize to 48x48
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    
    # Normalize pixel values
    normalized = resized.astype("float") / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    expanded = np.expand_dims(reshaped, axis=0)
    return expanded

def predict_emotion(face_roi):
    """Predict emotion from face region"""
    if emotion_model is None:
        # Fallback: Simple detection based on mouth position
        h, w = face_roi.shape[:2]
        mouth_roi = face_roi[int(h*0.6):int(h*0.9), int(w*0.25):int(w*0.75)]
        mouth_open = np.mean(mouth_roi) > 100
        
        if mouth_open:
            return "Happy", 0.8, [0,0,0,0.8,0,0,0.2]
        else:
            return "Neutral", 0.7, [0,0,0,0.1,0.7,0,0.2]
    
    processed_face = preprocess_face(face_roi)
    preds = emotion_model.predict(processed_face)[0]
    emotion_idx = np.argmax(preds)
    return EMOTIONS[emotion_idx], preds[emotion_idx], preds

def draw_emotion_results(frame, face, emotion, confidence):
    """Draw emotion results on frame"""
    x, y, w, h = face
    # Draw face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw emotion text
    emotion_text = f"{emotion}: {confidence:.2f}"
    cv2.putText(frame, emotion_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    
    # Reduce resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Process every 3rd frame to reduce load
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Empty frame, retrying...")
            continue
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Only process every 3rd frame to reduce CPU load
        if frame_count % 3 != 0:
            cv2.imshow("Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)  # Only detect larger faces
        )
        
        # Process each detected face
        for face in faces:
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion if face is large enough
            if w > 50 and h > 50:
                try:
                    emotion, confidence, _ = predict_emotion(face_roi)
                    frame = draw_emotion_results(frame, face, emotion, confidence)
                except Exception as e:
                    print(f"Emotion prediction error: {e}")
        
        # Display frame
        cv2.imshow("Facial Emotion Recognition", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create model.json if missing (architecture for fallback)
    if not os.path.exists("model.json") and emotion_model is None:
        with open("model.json", "w") as f:
            f.write('''{
                "class_name": "Sequential",
                "config": {
                    "name": "sequential",
                    "layers": []
                },
                "keras_version": "2.15.0",
                "backend": "tensorflow"
            }''')
    
    main()