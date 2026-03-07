import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import threading

# --- 1. INITIALIZATION ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 2. AUDIO DETECTION ENGINE ---
# Resolves the "Audio Not Detected" failure in your report
audio_level = 0
def audio_callback(indata, frames, time, status):
    global audio_level
    audio_level = np.linalg.norm(indata) * 10

def start_audio():
    with sd.InputStream(callback=audio_callback):
        while True: sd.sleep(1000)

threading.Thread(target=start_audio, daemon=True).start()

# --- 3. CONFIGURATION ---
POSTURE_MIN_GAP = 0.12  # Threshold for Hunching (Nose vs. Shoulder)
TILT_MAX_DIFF = 0.04    # Threshold for Leaning (Shoulder vs. Shoulder)
SMILE_MIN_LIFT = 0.03   # Threshold for Smiling (Lip corners)

cap = cv2.VideoCapture(0)

print("Aviation AI System Active. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Prep Image
    frame = cv2.flip(frame, 1)
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- STATUS VARIABLES ---
    posture_msg, p_color = "POSTURE: Analyzing...", (255, 255, 255)
    face_msg, f_color = "FACE: Analyzing...", (255, 255, 255)
    audio_msg, a_color = "AUDIO: No Sound", (0, 0, 255)

    # 1. POSTURE LOGIC (Rectified for Sitting/Standing/Tilting)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Hunch Detection (Vertical Gap)
        shoulder_center_y = (lm[11].y + lm[12].y) / 2
        nose_y = lm[0].y
        gap = shoulder_center_y - nose_y
        
        # Tilt Detection (Shoulder Symmetry)
        shoulder_diff = abs(lm[11].y - lm[12].y)

        if gap < POSTURE_MIN_GAP:
            posture_msg, p_color = "REJECT: Slouching/Hunching Detected", (0, 0, 255)
        elif shoulder_diff > TILT_MAX_DIFF:
            posture_msg, p_color = "REJECT: Tilted/Bended Posture", (0, 0, 255)
        else:
            posture_msg, p_color = "POSTURE: Professional (Straight)", (0, 255, 0)

    # 2. FACE LOGIC (Smile Detection)
    if results.face_landmarks:
        f = results.face_landmarks.landmark
        # Calculate if corners of mouth are higher than center (Smile)
        mouth_center_y = (f[13].y + f[14].y) / 2
        smile_lift = mouth_center_y - min(f[61].y, f[291].y)

        if smile_lift > SMILE_MIN_LIFT:
            face_msg, f_color = "FACE: Genuine Smile Accepted", (0, 255, 0)
        else:
            face_msg, f_color = "REJECT: Frown/Neutral (Smile Required)", (0, 0, 255)

    # 3. AUDIO LOGIC
    if audio_level > 0.4:
        audio_msg, a_color = "AUDIO: Sound Detected", (0, 255, 0)

    # --- UI DISPLAY ---
    # Background rectangle for readability
    cv2.rectangle(frame, (0,0), (600, 160), (0,0,0), -1)
    cv2.putText(frame, posture_msg, (10, 40), 1, 1.5, p_color, 2)
    cv2.putText(frame, face_msg, (10, 85), 1, 1.5, f_color, 2)
    cv2.putText(frame, audio_msg, (10, 130), 1, 1.5, a_color, 2)

    # Visualization
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    cv2.imshow('Final Aviation AI Behavior Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
