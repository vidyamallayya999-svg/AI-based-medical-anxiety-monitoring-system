
import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import csv
from pathlib import Path
import time
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Medical Anxiety Monitoring System", layout="wide")
st.markdown("""
<style>

/* Main Background */
[data-testid="stAppViewContainer"]{
    background-color: #EAF6FF;
}

/* Sidebar Background */
[data-testid="stSidebar"]{
    background-color: #D6EAF8;
}

</style>
""", unsafe_allow_html=True)

CSV_FILE = "anxiety_log.csv"
if not Path(CSV_FILE).exists():
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Patient Name",
            "Timestamp",
            "Alert",
            "Score",
            "Level"
        ])

st.markdown("""
<h1 style='
text-align:center;
color:#003366;
font-size:42px;
font-weight:bold;
'>
AI-Based Medical Anxiety Monitoring System
</h1>
""", unsafe_allow_html=True)

patient_name=st.text_input("Enter the name of the patient")
st.markdown("Real-time Anxiety Detection using MediaPipe + Streamlit")

procedure = st.sidebar.selectbox(
    "Select Procedure",
    ["Dental", "Ultrasound", "Injection"]
)

# Validation
if patient_name == "":
    st.warning("Please enter patient name")
    st.stop()

# Display Patient Information
col1, col2 = st.columns(2)

with col1:
    st.info(f"Patient: {patient_name}")

with col2:
    st.info(f"Procedure: {procedure}")

class AnxietyProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose()
        self.face = self.mp_face.FaceMesh(max_num_faces=1,refine_landmarks=True)
        self.hands = self.mp_hands.Hands()

        self.prev_nose_x = None
        self.head_move_count = 0
        self.blink_count = 0
        self.anxiety_score = 0
        self.last_alert = ""

        self.previous_left_knee_x = None
        self.previous_right_knee_x = None
        self.prev_wrist_x = None
        self.prev_wrist_y = None
        self.tremble_count = 0

        self.last_save_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose_result = self.pose.process(rgb)
        face_result = self.face.process(rgb)
        hand_result = self.hands.process(rgb)

        alerts = []

        if pose_result.pose_landmarks:
            nose = pose_result.pose_landmarks.landmark[
              self.mp_pose.PoseLandmark.NOSE
            ]

            nose_x = int(nose.x * w)

            if self.prev_nose_x is not None:

              movement = abs(nose_x - self.prev_nose_x)

              if movement > 5:
                self.head_move_count += 1

            if self.head_move_count > 8:
               alerts.append("Head Shaking")
               self.anxiety_score += 1
               self.head_move_count = 0

            self.prev_nose_x = nose_x

        if face_result.multi_face_landmarks:
            face = face_result.multi_face_landmarks[0]

            eye_gap = abs(face.landmark[159].y - face.landmark[145].y)

            if eye_gap < 0.008:
                self.blink_count += 1

            if self.blink_count > 15:
                alerts.append("Rapid Blinking")
                self.anxiety_score += 2
                self.blink_count = 0

            lip_gap = abs(face.landmark[13].y - face.landmark[14].y)

            if lip_gap < 0.004:
                alerts.append("Lip Pressing")
                self.anxiety_score += 2

            eye_center = (
             face.landmark[33].x +
             face.landmark[263].x
             ) / 2

            if eye_center < 0.40 or eye_center > 0.60:
               alerts.append("Looking Away")
               self.anxiety_score += 1

        if hand_result.multi_hand_landmarks:
          for hand in hand_result.multi_hand_landmarks:

              wrist = hand.landmark[0]
              wrist_x = int(wrist.x * w)
              wrist_y = int(wrist.y * h)

              if wrist.y < 0.7:

                if procedure == "Dental":
                  alerts.append("Sudden Hand Movement")
                  self.anxiety_score += 1

                elif procedure == "Ultrasound":
                  alerts.append("Possible Slapping Movement")
                  self.anxiety_score += 1

                elif procedure == "Injection":
                  alerts.append("Hand Withdrawal")

                  self.anxiety_score += 1

            # Hand Trembling

              if self.prev_wrist_x is not None:

                 movement = math.sqrt(
                 (wrist_x - self.prev_wrist_x) ** 2 +
                  (wrist_y - self.prev_wrist_y) ** 2
                )

                 if 3 < movement < 15:
                   self.tremble_count += 1
                 else:
                    self.tremble_count=0

                 if self.tremble_count > 20:
                  alerts.append("Hand Trembling")
                  self.anxiety_score += 2
                  self.tremble_count = 0

              self.prev_wrist_x = wrist_x
              self.prev_wrist_y = wrist_y

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark

            # left_knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
            # right_knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]

            # if self.previous_left_knee_x is not None:
            #    left_move = abs(left_knee.x - self.previous_left_knee_x)
            #    right_move = abs(right_knee.x - self.previous_right_knee_x)

            # if left_move > 0.08 or right_move > 0.08:
            #     alerts.append("Leg Kicking")
            #     self.anxiety_score += 2

            # if self.previous_left_knee_x is not None:
            #   if (
            #      abs(left_knee.x - self.previous_left_knee_x) > 0.15 or
            #      abs(right_knee.x - self.previous_right_knee_x) > 0.15
            #     ):
            #       alerts.append("Leg Pull Away")
            #       self.anxiety_score += 2

            # self.previous_left_knee_x = left_knee.x
            # self.previous_right_knee_x = right_knee.x

            left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            if abs(left_shoulder.z - right_shoulder.z) > 0.10:
               alerts.append("Body Twisting")
               self.anxiety_score += 2

            left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_center = (left_hip.y + right_hip.y) / 2

            if hip_center < 0.45:
              alerts.append("Attempting To Stand Up")
              self.anxiety_score += 3

        self.anxiety_score = min(self.anxiety_score, 20)

        if self.anxiety_score < 5:
            level = "LOW"
        elif self.anxiety_score < 10:
            level = "MODERATE"
        else:
            level = "HIGH"
        
        print(alerts)
        if alerts:

          current_time = time.time()

          if current_time - self.last_save_time > 2:

            with open(CSV_FILE, "a", newline="") as f:
              writer = csv.writer(f)

              for alert in set(alerts):   # save every unique alert
                  writer.writerow([
                     patient_name,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     alert,
                     self.anxiety_score,
                     level
                ])

            self.last_save_time = current_time

        for i, alert in enumerate(alerts):
            cv2.putText(
                img,
                alert,
                (20, 40 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            img,
            f"Score: {self.anxiety_score}",
            (20, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            img,
            f"Level: {level}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if st.button("Start Monitoring"):
    st.session_state.monitoring = True

if st.session_state.monitoring:
    st.subheader("Live Monitoring")

    left, center, right = st.columns([1,2,1])

    with center:
      webrtc_streamer(
          key="anxiety-monitor",
          video_processor_factory=AnxietyProcessor,
          media_stream_constraints={"video": True, "audio": False},
        )

st.subheader("Dashboard")

try:
    df = pd.read_csv(CSV_FILE)

    if not df.empty:
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Alerts", len(df))
        c2.metric("Maximum Score", int(df["Score"].max()))
        c3.metric("Latest Level", str(df["Level"].iloc[-1]))

        st.dataframe(df, use_container_width=True)

        chart_df = df.copy()
        chart_df["Timestamp"] = pd.to_datetime(chart_df["Timestamp"])

        st.line_chart(
            chart_df.set_index("Timestamp")["Score"]
        )

        csv_data = df.to_csv(index=False)

        st.download_button(
            "Download Anxiety Report",
            csv_data,  
            "anxiety_report.csv",
            "text/csv"
        )
except Exception as e:
    st.error(f"error reading CSV: {e}")