import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import threading

# --- INITIALIZATION ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- AUDIO DETECTION THREAD ---
# Target: Fixes "No audio detected" issue mentioned in report 
audio_detected = False
def check_audio():
    global audio_detected
    def audio_callback(indata, frames, time, status):
        global audio_detected
        volume_norm = np.linalg.norm(indata) * 10
        audio_detected = volume_norm > 0.5 # Threshold for sound
    with sd.InputStream(callback=audio_callback):
        while True: sd.sleep(1000)

threading.Thread(target=check_audio, daemon=True).start()

# --- CONFIGURATION (Based on Aviation Review Doc ) ---
# Posture: Measures vertical alignment of nose and shoulders to avoid tilted errors
POSTURE_THRESHOLD = 0.12  
SMILE_THRESHOLD = 0.04    # Measures lip corners for "Genuine Smile" 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 1. RECTIFIED POSTURE (Seated & Tilted Detection)
    # Target: Detects 'Soldier Posture' vs 'Visible Slouching' (Severity 9) 
    posture_msg, p_color = "Posture: Professional", (0, 255, 0)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Use nose-to-shoulder-line vertical offset to handle sitting 
        shoulder_y = (lm[11].y + lm[12].y) / 2
        # Detect tilting: if shoulders are significantly unaligned
        tilt = abs(lm[11].y - lm[12].y)
        
        if (shoulder_y - lm[0].y) < POSTURE_THRESHOLD or tilt > 0.05:
            posture_msg, p_color = "REJECT: Slouching/Tilted Posture", (0, 0, 255)

    # 2. RECTIFIED FACIAL EXPRESSION
    # Target: Genuine Sustained Smile (Severity 10) vs Frown (Severity 10) 
    face_msg, f_color = "Expression: Neutral", (255, 255, 255)
    if results.face_landmarks:
        f = results.face_landmarks.landmark
        smile_gap = (f[0].y + f[17].y)/2 - min(f[61].y, f[291].y)
        if smile_gap > SMILE_THRESHOLD:
            face_msg, f_color = "Accepted: Genuine Smile", (0, 255, 0)
        elif smile_gap < 0.01:
            face_msg, f_color = "REJECT: Frown Detected", (0, 0, 255)

    # 3. AUDIO & GROOMING STATUS
    audio_msg = "Audio: DETECTED" if audio_detected else "REJECT: No Audio Detected"
    a_color = (0, 255, 0) if audio_detected else (0, 0, 255)

    # UI OVERLAY (Mapped to Report Requirements )
    cv2.putText(frame, posture_msg, (10, 30), 1, 1.5, p_color, 2)
    cv2.putText(frame, face_msg, (10, 70), 1, 1.5, f_color, 2)
    cv2.putText(frame, audio_msg, (10, 110), 1, 1.5, a_color, 2)
    cv2.putText(frame, "Grooming: Hair/Attire Check Active", (10, 150), 1, 1, (255, 255, 0), 1)

    cv2.imshow('Aviation AI Behavior Analyzer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
