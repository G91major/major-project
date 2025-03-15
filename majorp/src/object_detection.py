import cv2
import numpy as np

# Load video
video_path = "C:/Users/harsh/Downloads/demo4.mp4"
cap = cv2.VideoCapture(video_path)

# Get frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Background subtractor
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

# Kernel for noise removal
kernel = np.ones((3, 3), np.uint8)

# Load Haar Cascade for person detection
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Tracking storage
tracked_objects = {}  # {id: (x, y, w, h, frames_since_seen)}
object_id = 0
max_disappearance_frames = 15  # Keep objects longer before removing
distance_threshold = 50  # Lower distance to avoid ID swaps

def assign_ids(detected_boxes):
    """Assigns unique IDs to detected objects and maintains consistency."""
    global object_id
    new_tracked_objects = {}

    for (x, y, w, h) in detected_boxes:
        matched_id = None
        min_distance = float('inf')

        for obj_id, (prev_x, prev_y, prev_w, prev_h, frames_since_seen) in tracked_objects.items():
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            if distance < distance_threshold and distance < min_distance:
                min_distance = distance
                matched_id = obj_id

        if matched_id is not None:
            new_tracked_objects[matched_id] = (x, y, w, h, 0)
        else:
            object_id += 1
            new_tracked_objects[object_id] = (x, y, w, h, 0)

    # Keep recently disappeared objects
    for obj_id, (x, y, w, h, frames_since_seen) in tracked_objects.items():
        if obj_id not in new_tracked_objects and frames_since_seen < max_disappearance_frames:
            new_tracked_objects[obj_id] = (x, y, w, h, frames_since_seen + 1)

    return new_tracked_objects

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction & noise removal
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection & contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes = []

    # Extract bounding boxes from contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if 2000 < area < 40000:  # Remove very small and large objects
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 1.2:  # Human-like aspect ratio
                detected_boxes.append((x, y, w, h))

    # Haar cascade person detection
    bodies = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 60))
    for (x, y, w, h) in bodies:
        detected_boxes.append((x, y, w, h))

    # Assign stable IDs
    tracked_objects = assign_ids(detected_boxes)

    # Draw tracked objects
    for obj_id, (x, y, w, h, _) in tracked_objects.items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save and show frame
    out.write(frame)
    cv2.imshow("Stable Tracking Without YOLO", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Stable Tracking Completed! Output saved as {output_path}")
