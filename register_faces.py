import cv2
import os
import json
import numpy as np
from insightface.app import FaceAnalysis

# ----- Setup -----
# Initialize InsightFace model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # Use -1 for CPU only

# Path for saving known faces
FACE_DB_PATH = "known_faces.json"

# Load or create known faces database
if os.path.exists(FACE_DB_PATH):
    with open(FACE_DB_PATH, "r") as f:
        known_faces = json.load(f)
else:
    known_faces = {}  # {"person_name": [embedding list]}

def normalize(vector):
    """
    Normalize an embedding vector to have unit length.
    """
    return vector / np.linalg.norm(vector)

def save_face(name, embedding):
    """
    Save a normalized face embedding with an associated name into the JSON file.
    """
    embedding = normalize(embedding).tolist()
    known_faces[name] = embedding
    with open(FACE_DB_PATH, "w") as f:
        json.dump(known_faces, f)
    print(f"Saved normalized embedding for {name}.")

def register_face_from_camera():
    """
    Register new faces using the webcam and save them into the known_faces.json file.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    print("Press 's' to save a face, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the camera.")
            break

        # Process frame with InsightFace to detect faces
        faces = face_app.get(frame)

        # Draw bounding boxes around detected faces
        for face in faces:
            (x, y, x2, y2) = face.bbox.astype(int)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 2)

        # Show video feed with detected faces
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)
        
        if key == ord('s') and faces:  # Save a face when 's' key is pressed
            print("Detected face(s).")
            name = input("Enter name for this face: ")
            if name.strip() == "":
                print("Name cannot be empty. Try again.")
                continue
            
            # Save embedding for the first detected face
            embedding = normalize(faces[0].embedding)
            save_face(name, embedding.tolist())
            print(f"Face registered for {name}: {embedding}")
        
        elif key == ord('q'):  # Quit when 'q' key is pressed
            print("Exiting face registration.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ----- Main -----
if __name__ == "__main__":
    print("Welcome to the Face Registration Tool!")
    print("This tool allows you to add known faces to the system.")
    register_face_from_camera()