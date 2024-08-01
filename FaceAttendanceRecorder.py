import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

# Function to load known faces
def load_known_faces(csv_file='face_encoding2.csv'):
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []
    known_face_depts = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                name = row[0]
                emp_id = row[1]
                dept = row[2]
                encoding = np.array([float(value) for value in row[3:] if value], dtype=float)
                if len(encoding) == 128:  # Ensure the encoding is 128-dimensional
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    known_face_ids.append(emp_id)
                    known_face_depts.append(dept)
                else:
                    print(f"Skipping row due to incorrect encoding length: {len(encoding)}")
            except ValueError as e:
                print(f"Skipping row due to error: {e}")
    return known_face_encodings, known_face_names, known_face_ids, known_face_depts

# Function to store attendance
def store_attendance(name, employee_id, dept, csv_file='attendance2.csv'):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, employee_id, dept, time])
    print(f"Recorded: {name}, {employee_id}, {dept}, {time}")

known_face_encodings, known_face_names, known_face_ids, known_face_depts = load_known_faces()

# Function to generate frames from webcam
def generate_frames():
    cap = cv2.VideoCapture(0)
    recorded_names = set()
    unknown_faces = []
    unknown_counter = 1

    print("Live camera opened")

    if not cap.isOpened():
        print("Error: Could not open video device. Change device.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_ids = []
        face_depts = []

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_distance = np.min(distances)
            tolerance = 0.4  # Adjust the tolerance as necessary
            if min_distance < tolerance:
                best_match_index = np.argmin(distances)
                name = known_face_names[best_match_index]
                emp_id = known_face_ids[best_match_index]
                dept = known_face_depts[best_match_index]
            else:
                unknown_match = False
                for i, unknown_face in enumerate(unknown_faces):
                    if face_recognition.compare_faces([unknown_face], face_encoding, tolerance=tolerance)[0]:
                        name = f"Unknown{i + 1}"
                        emp_id = "Unknown"
                        dept = "Unknown"
                        unknown_match = True
                        break
                
                if not unknown_match:
                    unknown_faces.append(face_encoding)
                    name = f"Unknown{unknown_counter}"
                    emp_id = "Unknown"
                    dept = "Unknown"
                    unknown_counter += 1

            face_names.append(name)
            face_ids.append(emp_id)
            face_depts.append(dept)

        for (top, right, bottom, left), name, emp_id, dept in zip(face_locations, face_names, face_ids, face_depts):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

            if name != "Unknown" and name not in recorded_names:
                store_attendance(name, emp_id, dept)
                recorded_names.add(name)
            elif "Unknown" in name and name not in recorded_names:
                store_attendance(name, emp_id, dept)
                recorded_names.add(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Live camera closed")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_frames()
