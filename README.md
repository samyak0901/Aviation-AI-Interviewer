Aviation AI Behavior Analyzer (CV Model)
Project Overview
This repository contains a Computer Vision (CV) algorithm designed to automate the evaluation of cabin crew candidates. Based on the IndiGo Cabin Crew recruitment standards, this model analyzes video feeds to detect high-priority behavioral traits and grooming standards

Objective
To build a scalable AI "Evaluation Engine" that identifies candidate rejections based on specific severity scores (1-10) defined in the Aviation Mock Interview grading rubric.

Features & Capabilities
Using MediaPipe and OpenCV, this prototype currently tracks:
1)Posture Analysis (Severity 10): Real-time detection of "Slouching" vs. "Professional Upright Posture" using skeletal landmark alignment.
2)Facial Expression Tracking: Monitoring the "Pan Am Smile" and detecting signs of "Emotional Leakage" (fear, panic, or irritation).
3)Grooming Check (POC): Baseline logic for identifying professional attire and facial grooming.
4)Eye Contact Monitoring: Gaze tracking to ensure the candidate is engaging with the interviewer/camera.

Tech Stack
1)Language: Python 3.9+
2)Libraries:  MediaPipe (Skeletal & Face Mesh tracking)
OpenCV (Video Processing)
NumPy (Mathematical alignment calculations)
3)Version Control: Git / GitHub

Project Structure
Plaintext
├── main.py              # Main execution script for real-time analysis
├── utils/
│   ├── posture.py       # Logic for calculating shoulder/hip angles
│   └── expressions.py   # Logic for mouth/eye landmark distances
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation

Roadmap
1)[x] Initial CV Pipeline setup (MediaPipe)
2)[x] Posture & Slouching Detection
3)[ ] Integration of YOLOv8 for Attire/Grooming detection
4)[ ] Exporting "Severity Scores" to a JSON format for the GreetAI backend

How to Run
1)Clone the repo: git clone https://github.com/YOUR_USERNAME/Aviation-AI.git
2)Install requirements: pip install -r requirements.txt
3)Run the analyzer: python main.py
Paste the text: Paste the content above.

The "Commit" message: When you save it, write: "docs: add professional project documentation and roadmap" (this makes you look very experienced).
