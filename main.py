!apt-get install -y portaudio19-dev
!pip install mediapipe sounddevice
import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import threading
from collections import deque
import json # Added for JSON export
import time # Added for FPS calculation

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
POSTURE_MIN_GAP = 0.12  # Threshold for Hunching (Nose vs. Shoulder vertical gap)
TILT_MAX_DIFF = 0.04    # Threshold for Leaning (Shoulder vs. Shoulder vertical difference)
FORWARD_HEAD_THRESHOLD = 0.03 # Threshold for Forward Head Posture (Ear vs. Shoulder horizontal offset)
SMILE_MIN_LIFT = 0.03   # Threshold for Smiling (Lip corners)
AUDIO_THRESHOLD = 0.4   # Minimum audio level to be considered 'sound detected'
IRIS_CENTER_THRESHOLD = 0.007 # A tighter threshold for direct gaze based on iris position

# --- Buffering for Stability ---
POSTURE_BUFFER_SIZE = 20 # Number of frames to average for posture (approx 0.6-0.7 seconds at 30 FPS)
SMILE_BUFFER_SIZE = 15   # Number of frames to average for smile
EYE_CONTACT_BUFFER_SIZE = 15 # Number of frames to average for eye contact
AUDIO_BUFFER_SIZE = 5    # Number of frames to average for audio

# History buffers for granular posture checks
posture_overall_good_history = deque(maxlen=POSTURE_BUFFER_SIZE)
hunching_history = deque(maxlen=POSTURE_BUFFER_SIZE)
tilting_history = deque(maxlen=POSTURE_BUFFER_SIZE)
forward_head_history = deque(maxlen=POSTURE_BUFFER_SIZE)

smile_history = deque(maxlen=SMILE_BUFFER_SIZE)
eye_contact_history = deque(maxlen=EYE_CONTACT_BUFFER_SIZE)
audio_history = deque(maxlen=AUDIO_BUFFER_SIZE)

# --- Global counters for session summary ---
total_frames_processed = 0
total_good_posture_frames = 0
total_hunching_frames = 0
total_tilting_frames = 0
total_forward_head_frames = 0
total_smiling_frames = 0
total_direct_gaze_frames = 0
total_audio_present_frames = 0

# --- 4. DETECTION FUNCTIONS ---
def detect_posture(pose_landmarks):
    is_hunching = False
    is_tilting = False
    is_forward_head = False
    is_overall_good = False
    
    # Initialize landmark coordinates for drawing
    shoulder_center_coords = None
    nose_coords = None
    left_shoulder_coords = None
    right_shoulder_coords = None
    left_ear_coords = None
    right_ear_coords = None

    if pose_landmarks:
        lm = pose_landmarks.landmark
        
        nose_coords = (int(lm[mp_holistic.PoseLandmark.NOSE].x * frame.shape[1]), int(lm[mp_holistic.PoseLandmark.NOSE].y * frame.shape[0]))
        left_shoulder_coords = (int(lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]), int(lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
        right_shoulder_coords = (int(lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]), int(lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
        shoulder_center_coords = (int((lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].x + lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x) / 2 * frame.shape[1]),
                                  int((lm[mp_holistic.PoseLandmark.LEFT_SHOULDER].y + lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y) / 2 * frame.shape[0]))
        left_ear_coords = (int(lm[mp_holistic.PoseLandmark.LEFT_EAR].x * frame.shape[1]), int(lm[mp_holistic.PoseLandmark.LEFT_EAR].y * frame.shape[0]))
        right_ear_coords = (int(lm[mp_holistic.PoseLandmark.RIGHT_EAR].x * frame.shape[1]), int(lm[mp_holistic.PoseLandmark.RIGHT_EAR].y * frame.shape[0]))


        # Hunch Detection (Vertical Gap: Nose vs. Shoulder Y-axis)
        shoulder_center_y = (lm[11].y + lm[12].y) / 2
        nose_y = lm[0].y
        vertical_gap = shoulder_center_y - nose_y
        if vertical_gap < POSTURE_MIN_GAP:
            is_hunching = True

        # Tilt Detection (Shoulder Symmetry)
        shoulder_diff = abs(lm[11].y - lm[12].y)
        if shoulder_diff > TILT_MAX_DIFF:
            is_tilting = True

        # Forward Head Posture Detection (Ear vs. Shoulder horizontal offset)
        if (lm[7].x - lm[11].x > FORWARD_HEAD_THRESHOLD) and \
           (lm[12].x - lm[8].x > FORWARD_HEAD_THRESHOLD):
            is_forward_head = True
            
        is_overall_good = not (is_hunching or is_tilting or is_forward_head)

    return is_overall_good, is_hunching, is_tilting, is_forward_head, \
           nose_coords, shoulder_center_coords, left_shoulder_coords, right_shoulder_coords, \
           left_ear_coords, right_ear_coords

def detect_face_expressions(face_landmarks, frame_shape):
    current_smile_detected = False
    current_eye_contact_direct = False
    
    left_iris_coords = None
    right_iris_coords = None
    left_eye_center_coords = None
    right_eye_center_coords = None

    if face_landmarks:
        f = face_landmarks.landmark

        # Smile Detection
        mouth_center_y = (f[13].y + f[14].y) / 2
        smile_lift = mouth_center_y - min(f[61].y, f[291].y)
        if smile_lift > SMILE_MIN_LIFT:
            current_smile_detected = True

        # Eye Contact Detection using iris landmarks
        # Check if iris landmarks are available (part of the full face mesh, > 477 landmarks)
        if len(f) > 477: 
            # User's Left Eye (appears on image Right): outer corner (133), inner corner (33), iris center (474)
            left_eye_outer_x = f[133].x
            left_eye_inner_x = f[33].x
            left_iris_x = f[474].x 

            # User's Right Eye (appears on image Left): outer corner (263), inner corner (362), iris center (468)
            right_eye_outer_x = f[263].x
            right_eye_inner_x = f[362].x
            right_iris_x = f[468].x 

            # Calculate the horizontal center of each eye
            left_eye_center_x = (left_eye_outer_x + left_eye_inner_x) / 2
            right_eye_center_x = (right_eye_outer_x + right_eye_inner_x) / 2

            # Check if iris is close to the center of the eye (small difference indicates direct gaze)
            if abs(left_iris_x - left_eye_center_x) < IRIS_CENTER_THRESHOLD and \
               abs(right_iris_x - right_eye_center_x) < IRIS_CENTER_THRESHOLD:
                current_eye_contact_direct = True
            
            left_iris_coords = (int(f[474].x * frame_shape[1]), int(f[474].y * frame_shape[0]))
            right_iris_coords = (int(f[468].x * frame_shape[1]), int(f[468].y * frame_shape[0]))
            left_eye_center_coords = (int(left_eye_center_x * frame_shape[1]), int(((f[133].y + f[33].y) / 2) * frame_shape[0]))
            right_eye_center_coords = (int(right_eye_center_x * frame_shape[1]), int(((f[263].y + f[362].y) / 2) * frame_shape[0]))

    return current_smile_detected, current_eye_contact_direct, \
           left_iris_coords, right_iris_coords, left_eye_center_coords, right_eye_center_coords

def detect_audio_presence(current_audio_level):
    return current_audio_level > AUDIO_THRESHOLD

def aggregate_and_display_status():
    posture_msg, p_color = "POSTURE: Analyzing...", (255, 255, 255)
    face_msg, f_color = "FACE: Analyzing...", (255, 255, 255)
    audio_msg, a_color = "AUDIO: No Sound", (0, 0, 255)
    eye_msg, e_color = "EYE CONTACT: Analyzing...", (255, 255, 255)

    # Posture Aggregation
    if len(posture_overall_good_history) == POSTURE_BUFFER_SIZE:
        num_good_posture = sum(posture_overall_good_history)
        if num_good_posture / POSTURE_BUFFER_SIZE > 0.7: # Majority are good
            posture_msg, p_color = "POSTURE: Professional (Straight)", (0, 255, 0)
        elif num_good_posture / POSTURE_BUFFER_SIZE < 0.3: # Majority are bad
            # Check recent granular issues for specific feedback
            if sum(hunching_history) / POSTURE_BUFFER_SIZE > 0.3: # If hunching is a frequent issue in the buffer
                posture_msg, p_color = "REJECT: Vertical Slouching/Hunching", (0, 0, 255)
            elif sum(tilting_history) / POSTURE_BUFFER_SIZE > 0.3: # If tilting is a frequent issue
                posture_msg, p_color = "REJECT: Tilted Posture Detected", (0, 0, 255)
            elif sum(forward_head_history) / POSTURE_BUFFER_SIZE > 0.3: # If forward head is a frequent issue
                posture_msg, p_color = "REJECT: Forward Head Posture", (0, 0, 255)
            else:
                posture_msg, p_color = "REJECT: Posture Issue Detected", (0, 0, 255)
        # else: Analyzing...

    # Face Aggregation (Smile)
    if len(smile_history) == SMILE_BUFFER_SIZE:
        num_smiles = sum(smile_history)
        if num_smiles / SMILE_BUFFER_SIZE > 0.7: # Majority are smiling
            face_msg, f_color = "FACE: Genuine Smile Accepted", (0, 255, 0)
        elif num_smiles / SMILE_BUFFER_SIZE < 0.3: # Majority are not smiling
            face_msg, f_color = "FACE: Neutral/Serious Detected", (0, 255, 0) # Neutral is okay
        # else: Analyzing...

    # Eye Contact Aggregation
    if len(eye_contact_history) == EYE_CONTACT_BUFFER_SIZE:
        num_direct_gaze = sum(eye_contact_history)
        if num_direct_gaze / EYE_CONTACT_BUFFER_SIZE > 0.7: # Majority are direct
            eye_msg, e_color = "EYE CONTACT: Direct Gaze Detected", (0, 255, 0)
        elif num_direct_gaze / EYE_CONTACT_BUFFER_SIZE < 0.3: # Majority are not direct
            eye_msg, e_color = "REJECT: Gaze Not Directed at Camera", (0, 0, 255)
        # else: Analyzing...

    # Audio Aggregation
    if len(audio_history) == AUDIO_BUFFER_SIZE:
        num_audio_present = sum(audio_history)
        if num_audio_present / AUDIO_BUFFER_SIZE > 0.5: # More than half have audio
            audio_msg, a_color = "AUDIO: Sound Detected", (0, 255, 0)
        else:
            audio_msg, a_color = "AUDIO: No Sound", (0, 0, 255)
            
    return posture_msg, p_color, face_msg, f_color, audio_msg, a_color, eye_msg, e_color


# --- 5. MAIN PROCESSING LOOP ---
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("ERROR: Could not open video stream. Check if camera is connected and not in use by another application.")
    exit()

print("Aviation AI System Active. Press 'q' to stop.")

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Prep Image and process with Holistic
    frame = cv2.flip(frame, 1) # Flip for selfie-view
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform Detections for Current Frame (and get landmark coordinates for drawing)
    (current_posture_overall_good, current_hunching, current_tilting, current_forward_head, 
     nose_coords, shoulder_center_coords, left_shoulder_coords, right_shoulder_coords, 
     left_ear_coords, right_ear_coords) = detect_posture(results.pose_landmarks)
    
    (current_smile_detected, current_eye_contact_direct, 
     left_iris_coords, right_iris_coords, left_eye_center_coords, right_eye_center_coords) = detect_face_expressions(results.face_landmarks, frame.shape)
    
    current_audio_present = detect_audio_presence(audio_level)

    # Update History Buffers
    posture_overall_good_history.append(current_posture_overall_good)
    hunching_history.append(current_hunching)
    tilting_history.append(current_tilting)
    forward_head_history.append(current_forward_head)

    smile_history.append(current_smile_detected)
    eye_contact_history.append(current_eye_contact_direct)
    audio_history.append(current_audio_present)

    # Update Global Counters for Session Summary
    total_frames_processed += 1
    if current_posture_overall_good: total_good_posture_frames += 1
    if current_hunching: total_hunching_frames += 1
    if current_tilting: total_tilting_frames += 1
    if current_forward_head: total_forward_head_frames += 1
    if current_smile_detected: total_smiling_frames += 1
    if current_eye_contact_direct: total_direct_gaze_frames += 1
    if current_audio_present: total_audio_present_frames += 1

    # Aggregate and get display messages
    posture_msg, p_color, face_msg, f_color, audio_msg, a_color, eye_msg, e_color = aggregate_and_display_status()

    # --- UI DISPLAY ---
    # Background rectangle for readability
    cv2.rectangle(frame, (0,0), (600, 190), (0,0,0), -1)
    cv2.putText(frame, posture_msg, (10, 40), 1, 1.5, p_color, 2)
    cv2.putText(frame, face_msg, (10, 85), 1, 1.5, f_color, 2)
    cv2.putText(frame, eye_msg, (10, 130), 1, 1.5, e_color, 2)
    cv2.putText(frame, audio_msg, (10, 175), 1, 1.5, a_color, 2)

    # Visualization
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Face mesh drawing confirmed.

    # --- Visual Feedback for Violations ---
    if current_hunching and nose_coords and shoulder_center_coords: # Hunching
        cv2.line(frame, nose_coords, shoulder_center_coords, (0, 0, 255), 2) # Red line for hunching
        cv2.circle(frame, nose_coords, 5, (0, 0, 255), -1)
        cv2.circle(frame, shoulder_center_coords, 5, (0, 0, 255), -1)

    if current_tilting and left_shoulder_coords and right_shoulder_coords: # Tilting
        cv2.line(frame, left_shoulder_coords, right_shoulder_coords, (0, 0, 255), 2) # Red line for tilt
        cv2.circle(frame, left_shoulder_coords, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_shoulder_coords, 5, (0, 0, 255), -1)

    if current_forward_head and left_ear_coords and right_ear_coords and left_shoulder_coords and right_shoulder_coords: # Forward Head
        cv2.line(frame, left_ear_coords, left_shoulder_coords, (0, 0, 255), 2) # Red lines for forward head
        cv2.line(frame, right_ear_coords, right_shoulder_coords, (0, 0, 255), 2)
        cv2.circle(frame, left_ear_coords, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_ear_coords, 5, (0, 0, 255), -1)

    if not current_eye_contact_direct and left_iris_coords and right_iris_coords and left_eye_center_coords and right_eye_center_coords: # Gaze not direct
        cv2.line(frame, left_iris_coords, left_eye_center_coords, (0, 165, 255), 2) # Orange line from iris to eye center
        cv2.line(frame, right_iris_coords, right_eye_center_coords, (0, 165, 255), 2)
        cv2.circle(frame, left_iris_coords, 3, (0, 165, 255), -1)
        cv2.circle(frame, right_iris_coords, 3, (0, 165, 255), -1)

    # Display FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 40), 1, 1.5, (255, 255, 255), 2)
    prev_frame_time = new_frame_time

    cv2.imshow('Final Aviation AI Behavior Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- 6. SESSION SUMMARY EXPORT (JSON) ---
session_summary = {}
if total_frames_processed > 0:
    session_summary = {
        "total_frames_processed": total_frames_processed,
        "overall_good_posture_percentage": (total_good_posture_frames / total_frames_processed) * 100,
        "hunching_detected_percentage": (total_hunching_frames / total_frames_processed) * 100,
        "tilting_detected_percentage": (total_tilting_frames / total_frames_processed) * 100,
        "forward_head_detected_percentage": (total_forward_head_frames / total_frames_processed) * 100,
        "overall_smiling_percentage": (total_smiling_frames / total_frames_processed) * 100,
        "overall_direct_gaze_percentage": (total_direct_gaze_frames / total_frames_processed) * 100,
        "overall_audio_present_percentage": (total_audio_present_frames / total_frames_processed) * 100
    }
    print("\n--- Session Summary (JSON) ---")
    print(json.dumps(session_summary, indent=4))

    # Export to JSON file
    try:
        with open('session_summary.json', 'w') as f:
            json.dump(session_summary, f, indent=4)
        print("Session summary saved to 'session_summary.json'")
    except IOError as e:
        print(f"Error saving session summary to file: {e}")
else:
    print("\n--- Session Summary ---")
    print("No frames processed during this session.")
