# =========================================
# AI-Powered Full-Body Virtual Try-On
# Streamlit App
# =========================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Virtual Try-On", page_icon="👗", layout="centered")
st.title("👗 AI-Powered Full-Body Virtual Try-On")

# -------------------------------
# Sidebar Upload
# -------------------------------
st.sidebar.header("Upload Images")
person_file = st.sidebar.file_uploader("Upload Person Image", type=["jpg", "jpeg", "png"])
shirt_file = st.sidebar.file_uploader("Upload Shirt PNG (transparent)", type=["png"])
pants_file = st.sidebar.file_uploader("Upload Pants PNG (transparent)", type=["png"])

# -------------------------------
# Helper Functions
# -------------------------------
def load_person_image(image_file):
    """Load person image as RGB"""
    return np.array(Image.open(image_file).convert("RGB"))

def load_garment_image(image_file):
    """Load garment image as RGBA (with transparency)"""
    return np.array(Image.open(image_file).convert("RGBA"))

def overlay_transparent(background, overlay, x, y):
    """Overlay PNG with alpha channel at position x, y"""
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if x+ol_w>bg_w or y+ol_h>bg_h:
        ol_w = min(ol_w, bg_w-x)
        ol_h = min(ol_h, bg_h-y)
        overlay = cv2.resize(overlay,(ol_w, ol_h))
        ol_h, ol_w = overlay.shape[:2]

    r, g, b, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((r,g,b))
    alpha = a/255.0

    bg_region = background[y:y+ol_h, x:x+ol_w]
    for c in range(3):
        bg_region[:,:,c] = alpha*overlay_rgb[:,:,c] + (1-alpha)*bg_region[:,:,c]
    background[y:y+ol_h, x:x+ol_w] = bg_region
    return background

def overlay_scaled_transparent(background, overlay, x, y, target_width, target_height):
    """Resize overlay and apply alpha blending"""
    overlay_resized = cv2.resize(overlay, (target_width, target_height))
    return overlay_transparent(background, overlay_resized, x, y)

# -------------------------------
# Main Logic
# -------------------------------
if person_file and (shirt_file or pants_file):
    person_img = load_person_image(person_file)
    img_h, img_w, _ = person_img.shape

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True)
    results = pose_detector.process(person_img)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # -------- Shirt Positioning --------
        if shirt_file:
            shirt_img = load_garment_image(shirt_file)
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_width = int(abs(left_shoulder.x - right_shoulder.x) * img_w)
            top_y = int(min(left_shoulder.y, right_shoulder.y) * img_h)
            top_x = int(left_shoulder.x * img_w)
            shirt_width = int(shoulder_width * 1.3)
            shirt_height = int(shirt_width * shirt_img.shape[0]/shirt_img.shape[1])
            person_img = overlay_scaled_transparent(person_img, shirt_img, top_x, top_y, shirt_width, shirt_height)

        # -------- Pants Positioning --------
        if pants_file:
            pants_img = load_garment_image(pants_file)
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Width: hip distance scaled
            hip_width = int(abs(left_hip.x - right_hip.x) * img_w * 1.1)
            # Height: hip → ankle distance
            leg_height = int(abs((left_hip.y + right_hip.y)/2 - (left_ankle.y + right_ankle.y)/2) * img_h * 1.05)

            pants_x = int(min(left_hip.x, right_hip.x) * img_w)
            pants_y = int(min(left_hip.y, right_hip.y) * img_h)
            person_img = overlay_scaled_transparent(person_img, pants_img, pants_x, pants_y, hip_width, leg_height)

    st.image(person_img, channels="RGB", caption="Virtual Try-On Result")



    
