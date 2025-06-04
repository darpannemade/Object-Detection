# UAV Real-time and Upload Video Tracker with Export Support

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
from collections import defaultdict
import time
import pygame
import io

# Initialize audio alert
pygame.mixer.init()
ALERT_SOUND_PATH = "C:\\Users\\Darpan\\Downloads\\Music\\alert.wav"
if os.path.exists(ALERT_SOUND_PATH):
    pygame.mixer.music.load(ALERT_SOUND_PATH)
ALERT_COOLDOWN = 3

# Load YOLOv8 model
model_path = "yolov8l.pt"
model = YOLO(model_path)

st.title("UAV Object Detection, Tracking, and No-Fly Zone Monitoring")

# Sidebar configuration
source_option = st.sidebar.selectbox("Choose Input Type", ["Upload Video", "Webcam (Real-time)"])
tracker_option = st.sidebar.selectbox("Tracking Algorithm", ["ByteTrack", "None"])
thermal_mode = st.sidebar.checkbox("Enable Thermal View (Simulated)")
draw_paths = st.sidebar.checkbox("Draw UAV Paths", value=True)

with st.sidebar.expander("No-Fly Zone Settings"):
    enable_zone = st.checkbox("Enable No-Fly Zone Detection", value=True)
    zone_x1 = st.number_input("Zone X1 (Top-Left)", min_value=0, value=200)
    zone_y1 = st.number_input("Zone Y1 (Top-Left)", min_value=0, value=100)
    zone_x2 = st.number_input("Zone X2 (Bottom-Right)", min_value=0, value=400)
    zone_y2 = st.number_input("Zone Y2 (Bottom-Right)", min_value=0, value=300)

def apply_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def process_video(video_path, is_webcam=False):
    args = Namespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        min_box_area=10,
        mot20=False
    )
    tracker = BYTETracker(args) if tracker_option == "ByteTrack" else None

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0) if not is_webcam and total_frames > 0 else None

    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    all_detections = []
    violations = []
    track_paths = defaultdict(list)
    frame_idx = 0
    last_alert_time = 0

    stop_btn_key = f"stop_btn_{int(time.time()*1000)}"
    stop_triggered = False
    if is_webcam:
        if "stop_webcam" not in st.session_state:
            st.session_state["stop_webcam"] = False
        if st.button("🛑 Stop and Save Results", key=stop_btn_key):
            st.session_state["stop_webcam"] = True
        stop_triggered = st.session_state.get("stop_webcam", False)

    while cap.isOpened():
        if is_webcam and stop_triggered:
            break

        success, frame = cap.read()
        if not success:
            break

        orig_frame = frame.copy()
        results = model.predict(source=orig_frame, save=False, conf=0.3, verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        if tracker:
            dets = np.concatenate([boxes[:, :4], confs.reshape(-1, 1)], axis=1)
            tracks = tracker.update(dets, (height, width), (height, width))

            for track in tracks:
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id
                class_id = class_ids[0] if len(class_ids) > 0 else -1
                conf = float(track.score) if hasattr(track, 'score') else 0.0

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                track_paths[track_id].append((cx, cy))

                in_zone = (zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2)
                if enable_zone and in_zone:
                    cv2.putText(annotated, "ZONE VIOLATION!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    violations.append([frame_idx, track_id, class_id, cx, cy])

                    if os.path.exists(ALERT_SOUND_PATH) and time.time() - last_alert_time > ALERT_COOLDOWN:
                        pygame.mixer.music.play()
                        last_alert_time = time.time()

                if draw_paths and len(track_paths[track_id]) > 1:
                    for i in range(1, len(track_paths[track_id])):
                        pt1 = track_paths[track_id][i - 1]
                        pt2 = track_paths[track_id][i]
                        cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID: {track_id}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                all_detections.append([frame_idx, track_id, class_id, conf, x1, y1, x2, y2])

        if enable_zone:
            cv2.rectangle(annotated, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)

        display_frame = apply_thermal(annotated.copy()) if thermal_mode else annotated.copy()
        out.write(display_frame)
        stframe.image(display_frame, channels="BGR", use_container_width=True)

        frame_idx += 1
        if not is_webcam and progress:
            progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    if not is_webcam and progress:
        progress.empty()

    if not is_webcam or stop_triggered:
        if all_detections:
            df = pd.DataFrame(all_detections, columns=["frame", "id", "class", "conf", "x1", "y1", "x2", "y2"])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Detections CSV", csv_buffer.getvalue(), file_name="detections.csv", mime="text/csv")

        if violations:
            vdf = pd.DataFrame(violations, columns=["frame", "id", "class", "cx", "cy"])
            csv_bytes = io.StringIO()
            vdf.to_csv(csv_bytes, index=False)
            st.download_button("📥 Download Zone Violations CSV", csv_bytes.getvalue(), file_name="violations.csv", mime="text/csv")

        with open(temp_video_path, "rb") as f:
            st.download_button("📥 Download Annotated Video", f.read(), file_name="annotated_output.avi", mime="video/avi")

# Input source handler
if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        process_video(tfile.name)
        os.remove(tfile.name)
else:
    st.warning("Real-time webcam support requires a local Streamlit run.")
    if st.button("Start Webcam"):
        process_video(0, is_webcam=True)
