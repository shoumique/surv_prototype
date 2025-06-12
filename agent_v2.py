# Smart Home Surveillance Prototype
# Uses YOLOv8 for object detection and InsightFace for face recognition

import cv2
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import numpy as np
import os
import json

# ----- Setup -----
# Load YOLOv8 model (nano for speed)
yolo_model = YOLO("yolov8n.pt")

# Load InsightFace model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # use -1 for CPU only

# Load known face database
FACE_DB_PATH = "known_faces.json"
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

def compare_faces(embedding, threshold=1.2):
    """
    Compare a given face embedding against known face embeddings.
    Returns the name of the matched person or 'Unknown'.
    """
    min_dist = float("inf")
    matched_name = "Unknown"

    # Ensure embedding is normalized
    embedding = normalize(embedding)

    for name, known_embedding in known_faces.items():
        known_embedding = normalize(np.array(known_embedding))

        # Compute distance between embeddings (Euclidean distance)
        dist = np.linalg.norm(known_embedding - embedding)
        print(f"Comparing with {name}: Distance = {dist}")

        if dist < min_dist and dist < threshold:
            min_dist = dist
            matched_name = name

    print(f"Match result: {matched_name} with distance = {min_dist}")
    return matched_name

def log_event(name):
    """
    Log the detected person's name and timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Detected: {name}\n"
    with open("event_log.txt", "a") as log_file:
        log_file.write(log_entry)
    print(log_entry.strip())

# ----- Webcam Surveillance Loop -----
cap = cv2.VideoCapture(0)
print("Running surveillance. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect with YOLOv8
    results = yolo_model(frame)[0]
    person_boxes = [b for b in results.boxes.data if int(b[5]) == 0]  # class 0 = person

    for box in person_boxes:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop the face region from the frame
        face_crop = frame[y1:y2, x1:x2]

        # Face recognition
        faces = face_app.get(face_crop)
        if faces:
            face = faces[0]

            # Normalize the generated embedding
            embedding = normalize(face.embedding)
            print("Generated Embedding:", embedding)

            # Compare against known faces
            name = compare_faces(embedding, threshold=1.2)  # Adjust threshold if needed
            log_event(name)

            # Draw bounding box and label around the detected face
            label = f"{name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the surveillance feed
    cv2.imshow("Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()