import cv2
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = ["Praveen Kumar"]

tracked_faces = {}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    new_tracked_faces = {}
    assigned_names = set()

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        
        matched_id = None
        for face_id, (prev_center, assigned_name) in tracked_faces.items():
            if abs(prev_center[0] - center[0]) < 50 and abs(prev_center[1] - center[1]) < 50:
                matched_id = face_id
                break

        if matched_id is not None:
            name = tracked_faces[matched_id][1]
        else:
            available_names = [n for n in names if n not in assigned_names]
            name = random.choice(available_names) if available_names else f"Person {len(tracked_faces) + 1}"
            matched_id = len(tracked_faces) + 1

        assigned_names.add(name)
        new_tracked_faces[matched_id] = (center, name)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 255, 10), 3)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 255, 10), 2)

    tracked_faces = new_tracked_faces

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()