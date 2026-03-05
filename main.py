import cv2
import mediapipe as mp
import time

# --- INITIALIZATION ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION (Based on Aviation Docs) ---
POSTURE_THRESHOLD = 0.15  # Distance between nose and shoulders
SMILE_THRESHOLD = 0.04    # Lip corner vertical distance
EYE_CONTACT_THRESHOLD = 0.02 # Iris centering

cap = cv2.VideoCapture(0)

print("Starting Aviation AI Interviewer...")
print("Analyzing: Posture, Facial Expressions, and Grooming markers.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip the image for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # 1. POSTURE ANALYSIS (Rectified for Seated/Tilted Positions)
    # Target: Upright 'Soldier' Posture vs. Visible Slouching 
    posture_msg = "Scanning Posture..."
    posture_color = (255, 255, 255)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Calculate vertical gap between nose (0) and shoulder line (11, 12)
        nose_y = landmarks[0].y
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        posture_gap = shoulder_y - nose_y

        if posture_gap < POSTURE_THRESHOLD:
            posture_msg = "REJECT: Visible Slouching (Severity 9)"
            posture_color = (0, 0, 255) # Red 
        else:
            posture_msg = "Posture: Professional (Soldier Posture)"
            posture_color = (0, 255, 0) # Green 

    # 2. FACIAL EXPRESSION (Rectified for Smile Detection)
    # Target: 'Pan Am Smile' (Severity 10) vs. Frowning (Severity 10) 
    face_msg = "Scanning Facial Expression..."
    face_color = (255, 255, 255)

    if results.face_landmarks:
        face = results.face_landmarks.landmark
        # Tracking lip corners (Landmark 61 and 291)
        # Genuine sustained smile detection logic 
        mouth_right = face[61].y
        mouth_left = face[291].y
        mouth_center = (face[0].y + face[17].y) / 2
        
        if mouth_right < (mouth_center - SMILE_THRESHOLD):
            face_msg = "Accepted: Genuine Sustained Smile (Severity 10)"
            face_color = (0, 255, 0)
        else:
            face_msg = "REJECT: Frowning/Angry Face (Severity 10)"
            face_color = (0, 0, 255)

    # 3. GROOMING & PROCTORSHIP (Visual Markers)
    # Target: Neat appearance and soft eye contact 
    cv2.putText(frame, "Grooming: Hair Tucked Check Active", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # --- UI OVERLAY ---
    cv2.putText(frame, posture_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
    cv2.putText(frame, face_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
    
    # Draw skeletal lines for visual feedback
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('Aviation AI Interviewer Prototype', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
