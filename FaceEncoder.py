import cv2
import face_recognition
import numpy as np
import csv

# Function to capture face encoding and save to CSV
def capture_face_encoding(csv_file='face_encoding2.csv'):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    name = input("Enter your name: ")
    emp_id = input("Enter your employee ID: ")
    dept = input("Enter your department: ")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                face_encoding = face_encodings[0]
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, emp_id, dept] + face_encoding.tolist())
                print(f"Face encoding for {name} has been saved.")
                break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_face_encoding()
