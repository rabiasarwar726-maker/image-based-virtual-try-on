import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="AI Virtual Try-On", page_icon="👗", layout="centered")
st.title("👗 AI-Powered Full-Body Virtual Try-On")

# Sidebar uploads
st.sidebar.header("Upload Images")
person_file = st.sidebar.file_uploader("Person Image", type=["jpg","jpeg","png"])
shirt_file = st.sidebar.file_uploader("Shirt PNG (transparent)", type=["png"])
pants_file = st.sidebar.file_uploader("Pants PNG (transparent)", type=["png"])

# ---------------- Helper Functions ----------------
def load_person_image(file):
    return np.array(Image.open(file).convert("RGB"))

def load_garment_image(file):
    return np.array(Image.open(file).convert("RGBA"))

def overlay_transparent(bg, fg, x, y):
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]
    if x+fw>bw or y+fh>bh:
        fw = min(fw, bw-x)
        fh = min(fh, bh-y)
        fg = cv2.resize(fg,(fw, fh))
    r,g,b,a = cv2.split(fg)
    fg_rgb = cv2.merge((r,g,b))
    alpha = a/255.0
    region = bg[y:y+fh, x:x+fw]
    for c in range(3):
        region[:,:,c] = alpha*fg_rgb[:,:,c] + (1-alpha)*region[:,:,c]
    bg[y:y+fh, x:x+fw] = region
    return bg

def overlay_scaled(bg, fg, x, y, w, h):
    fg_resized = cv2.resize(fg,(w,h))
    return overlay_transparent(bg, fg_resized, x, y)

# ---------------- Caching MediaPipe Model ----------------
@st.cache_resource
def get_pose_detector():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=True)

pose_detector = get_pose_detector()

# ---------------- Main Logic ----------------
if person_file and (shirt_file or pants_file):
    person_img = load_person_image(person_file)
    
    # Resize large images for faster processing
    max_dim = 800
    h, w = person_img.shape[:2]
    scale = max(h, w)/max_dim
    if scale > 1:
        person_img = cv2.resize(person_img, (int(w/scale), int(h/scale)))
    
    img_h, img_w, _ = person_img.shape
    results = pose_detector.process(person_img)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # -------- Shirt --------
        if shirt_file:
            shirt = load_garment_image(shirt_file)
            left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_width = int(abs(left_shoulder.x - right_shoulder.x)*img_w)
            top_y = int(min(left_shoulder.y,right_shoulder.y)*img_h)
            top_x = int(left_shoulder.x*img_w)
            sw = int(shoulder_width*1.3)
            sh = int(sw * shirt.shape[0]/shirt.shape[1])
            person_img = overlay_scaled(person_img, shirt, top_x, top_y, sw, sh)

        # -------- Pants --------
        if pants_file:
            pants = load_garment_image(pants_file)
            left_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
            hip_width = int(abs(left_hip.x-right_hip.x)*img_w*1.1)
            leg_height = int(abs((left_hip.y+right_hip.y)/2 - (left_ankle.y+right_ankle.y)/2)*img_h*1.05)
            px = int(min(left_hip.x,right_hip.x)*img_w)
            py = int(min(left_hip.y,right_hip.y)*img_h)
            person_img = overlay_scaled(person_img, pants, px, py, hip_width, leg_height)

    st.image(person_img, channels="RGB", caption="Virtual Try-On Result")
