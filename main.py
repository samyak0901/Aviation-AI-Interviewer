import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

print("AI Interviewer Started. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # 1. POSTURE CHECK (Logic: Shoulder height relative to nose)
    posture_status = "Scanning..."
    color = (255, 255, 255) # White
    
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        nose_y = landmarks[0].y
        avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        
        # Detection for Slouching (Higher Y value means lower on screen)
        if avg_shoulder_y > 0.6: 
            posture_status = "REJECT: Slouching Detected"
            color = (0, 0, 255) # Red
        else:
            posture_status = "Posture: Professional"
            color = (0, 255, 0) # Green

    # 2. SMILE CHECK (Logic: Distance between lip corners)
    if face_results.multi_face_landmarks:
        # Landmarks for lip corners (61 and 291)
        # In a real model, we'd calculate the Euclidean distance
        cv2.putText(frame, "Face Tracking Active", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display results
    cv2.putText(frame, f"Analysis: {posture_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow('Aviation AI Interviewer (Solo POC)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
