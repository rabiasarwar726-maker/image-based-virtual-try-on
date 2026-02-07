import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👗",
    layout="centered"
)

st.title("👗 AI-Based Virtual Try-On System")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload Images")
person_file = st.sidebar.file_uploader("Person Image", ["jpg", "jpeg", "png"])
shirt_file = st.sidebar.file_uploader("Shirt PNG (Transparent)", ["png"])
pants_file = st.sidebar.file_uploader("Pants PNG (Transparent)", ["png"])
debug = st.sidebar.checkbox("🔍 Show Pose Landmarks")
show_boxes = st.sidebar.checkbox("📏 Show Garment Bounding Boxes")

# ---------------- Helper Functions ----------------
def load_person_image(file):
    return np.array(Image.open(file).convert("RGB"))

def load_garment_image(file):
    return np.array(Image.open(file).convert("RGBA"))

def overlay_transparent(bg, fg, x, y):
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]
    if x >= bw or y >= bh:
        return bg

    fw = min(fw, bw - x)
    fh = min(fh, bh - y)
    fg = fg[:fh, :fw]

    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+fh, x:x+fw, c] = (
            alpha * fg[:, :, c] + (1 - alpha) * bg[y:y+fh, x:x+fw, c]
        )
    return bg

def overlay_scaled(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))
    return overlay_transparent(bg, fg, x, y)

# ---------------- MediaPipe ----------------
@st.cache_resource
def load_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )

pose = load_pose()
mp_pose = mp.solutions.pose

# ---------------- Main Logic ----------------
if person_file and pose:
    person_img = load_person_image(person_file)

    max_dim = 900
    h, w = person_img.shape[:2]
    scale = max(h, w) / max_dim
    if scale > 1:
        person_img = cv2.resize(person_img, (int(w/scale), int(h/scale)))

    img_h, img_w, _ = person_img.shape
    results = pose.process(person_img)

    if not results.pose_landmarks:
        st.error("❌ No person detected. Use a clear front-facing image.")
        st.stop()

    lm = results.pose_landmarks.landmark

    if debug:
        debug_img = person_img.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            debug_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        st.image(debug_img, caption="Pose Debug View", use_container_width=True)

    # ---------------- SHIRT (Neck → Hip) ----------------
    if shirt_file:
        shirt = load_garment_image(shirt_file)

        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        neck_x = int((ls.x + rs.x) / 2 * img_w)
        neck_y = int(min(ls.y, rs.y) * img_h - 20)

        shoulder_width = abs(rs.x - ls.x) * img_w
        shirt_width = int(shoulder_width * 1.8)

        hip_y = int((lh.y + rh.y) / 2 * img_h)
        shirt_height = int((hip_y - neck_y) * 1.05)

        x = int(neck_x - shirt_width / 2)
        y = neck_y

        if show_boxes:
            dbg = person_img.copy()
            cv2.rectangle(dbg, (x, y), (x+shirt_width, y+shirt_height), (0,255,0), 2)
            st.image(dbg, caption="Shirt Box (Neck → Hip)", use_container_width=True)

        person_img = overlay_scaled(person_img, shirt, x, y, shirt_width, shirt_height)

    # ---------------- PANTS (Hip → Ankles, Cover Both Legs) ----------------
    if pants_file:
        pants = load_garment_image(pants_file)

        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        hip_y = int(min(l_hip.y, r_hip.y) * img_h)
        ankle_y = int(max(l_ankle.y, r_ankle.y) * img_h)

        center_x = int((l_hip.x + r_hip.x) / 2 * img_w)

        hip_width = abs(r_hip.x - l_hip.x) * img_w
        pant_width = int(hip_width * 2.2)
        pant_height = int((ankle_y - hip_y) * 1.05)

        x = int(center_x - pant_width / 2)
        y = hip_y

        if show_boxes:
            debug_box = person_img.copy()
            cv2.rectangle(
                debug_box,
                (x, y),
                (x + pant_width, y + pant_height),
                (0, 0, 255),
                2
            )
            st.image(
                debug_box,
                caption="👖 Pants Bounding Box (Hip → Ankles)",
                channels="RGB",
                use_container_width=True
            )

        pants_resized = cv2.resize(pants, (pant_width, pant_height))
        person_img = overlay_transparent(person_img, pants_resized, x, y)

    # ---------------- FINAL OUTPUT ----------------
    st.image(
        person_img,
        caption="✅ Virtual Try-On Result",
        channels="RGB",
        use_container_width=True
    )
