import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
kernel = np.ones((3, 3), np.uint8)

# Load Haar Cascades
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Tracking dictionary {ID: (x, y, w, h, frames_since_seen)}
tracked_objects = {}
object_id = 0
max_disappearance_frames = 30  # Keep tracking even when still

def assign_ids(detected_boxes):
    """ Assigns IDs to tracked objects, keeping them stable even when still. """
    global object_id
    new_tracked_objects = {}

    for (x, y, w, h) in detected_boxes:
        matched_id = None
        min_distance = float('inf')

        for obj_id, (prev_x, prev_y, prev_w, prev_h, frames_since_seen) in tracked_objects.items():
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            if distance < 70:  # Closer means likely the same object
                if distance < min_distance:
                    min_distance = distance
                    matched_id = obj_id

        if matched_id is not None:
            new_tracked_objects[matched_id] = (x, y, w, h, 0)  # Reset disappearance counter
        else:
            object_id += 1
            new_tracked_objects[object_id] = (x, y, w, h, 0)  # New object

    # Retain objects that disappeared recently
    for obj_id, (x, y, w, h, frames_since_seen) in tracked_objects.items():
        if obj_id not in new_tracked_objects and frames_since_seen < max_disappearance_frames:
            new_tracked_objects[obj_id] = (x, y, w, h, frames_since_seen + 1)

    return new_tracked_objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction & noise removal
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection & contours
    edges = cv2.Canny(fgmask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes = []

    # Bounding boxes for detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1500 < area < 30000:  # Filter by object size
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 1.2:  # Human-like shape
                detected_boxes.append((x, y, w, h))

    # Haar cascade detections
    bodies = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 100))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # **Fix: Ensure both arrays are valid before merging**
    for (x, y, w, h) in list(bodies) + list(faces):  
        detected_boxes.append((x, y, w, h))

    # Assign stable IDs
    tracked_objects = assign_ids(detected_boxes)

    # Draw tracked objects
    for obj_id, (x, y, w, h, _) in tracked_objects.items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show tracking output
    cv2.imshow("Real-Time Stable Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
