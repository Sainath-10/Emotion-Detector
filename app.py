import cv2
import numpy as np
import tensorflow as tf

# Try to load tensorflow_hub in case model.h5 contains a Hub layer (like ResNet v2 from TF Hub)
try:
    import tensorflow_hub as hub
    custom_objects = {'KerasLayer': hub.KerasLayer}
except ImportError:
    custom_objects = {}

# Emotion labels in standard alphabetical order (corresponding to Keras directory generator order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_emotion_model(model_path='model.h5'):
    print(f"Loading model from '{model_path}'...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {e}")
        print("Attempting to load model without custom objects...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model

def main():
    # Load the trained model
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        # Fallback to model_acc_61.42.h5 if model.h5 doesn't exist
        fallback_path = 'model_acc_61.42.h5'
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            raise FileNotFoundError("Neither 'model.h5' nor 'model_acc_61.42.h5' was found in the directory.")
            
    model = load_emotion_model(model_path)
    
    # Load face detection classifier (Haar Cascade frontal face default)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade face detector.")

    # Initialize webcam capture
    print("Opening webcam stream... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Please verify connection and permissions.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        # Flip the frame horizontally for a more natural mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert frame to grayscale for Haar Cascade face detector
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Crop the face from the BGR frame
            face_roi = frame[y:y+h, x:x+w]
            
            # Resize cropped face to match model's expected input shape (48x48)
            face_resized = cv2.resize(face_roi, (48, 48))
            
            # Normalize face crop pixels to [0, 1] range matching the training rescale configuration
            face_normalized = face_resized / 255.0
            
            # Add batch dimension: shape becomes (1, 48, 48, 3)
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Perform prediction
            predictions = model.predict(face_batch, verbose=0)
            max_index = np.argmax(predictions[0])
            confidence = predictions[0][max_index]
            predicted_emotion = EMOTION_LABELS[max_index]
            
            # Display label and draw bounding box
            label_text = f"{predicted_emotion} ({confidence * 100:.1f}%)"
            
            # Use different bounding box colors for positive vs negative/neutral emotions
            if predicted_emotion == 'Happy':
                box_color = (0, 255, 0) # Green for Happy
            elif predicted_emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
                box_color = (0, 0, 255) # Red for negative emotions
            else:
                box_color = (255, 255, 0) # Cyan for Neutral/Surprise
                
            # Draw bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Put label text above the bounding box
            cv2.putText(
                frame, 
                label_text, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                box_color, 
                2, 
                cv2.LINE_AA
            )
            
        # Display the output screen window
        cv2.imshow('Real-time Facial Emotion Recognition', frame)
        
        # Break frame loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up and release camera stream
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream closed.")

if __name__ == '__main__':
    import os
    main()
