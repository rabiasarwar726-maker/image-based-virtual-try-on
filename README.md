Virtual Try-On System (Image-Based)

An AI-powered virtual try-on system that overlays a shirt and pants onto a person’s image using human pose estimation.
The system automatically aligns garments based on neck, shoulders, hips, waist, and legs for realistic placement.

🚀 Features

Upload a front-facing person image

Automatic pose detection using MediaPipe

Shirt alignment:

Starts from neck

Ends at hip

Width based on shoulder distance

Pants alignment:

Starts from waist/hip

Ends at ankles

Width based on hip size

Transparent PNG garment support

Web interface using Streamlit

Docker-ready for deployment on Hugging Face Spaces

🧠 Technologies Used

Python

MediaPipe (Pose Estimation)

OpenCV

NumPy

Streamlit

Docker

📂 Project Structure
virtual-try-on/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── clothes/
│   ├── shirt.png          # Shirt PNG (transparent)
│   └── pants.png          # Pants PNG (transparent)
└── README.md

🖼️ How It Works

User uploads a person image

MediaPipe detects body landmarks

Shirt is resized and placed:

From neck → hip

Using shoulder width

Pants are resized and placed:

From waist → ankle

Using hip width

Garments are overlaid using alpha blending

Final virtual try-on image is displayed

▶️ Run Locally
1️⃣ Clone the repository
git clone https://github.com/rabiasarwar726-maker/virtual-try-on.git
cd virtual-try-on

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

🐳 Docker Deployment (Hugging Face)

This project is fully compatible with Hugging Face Spaces (Docker).

Steps:

Create a new Space

Select Docker as the SDK

Upload:

app.py

Dockerfile

requirements.txt

clothes/ folder

Hugging Face will automatically build and deploy the app

⚠️ Requirements & Notes

Use front-facing images for best results

Garment images must be PNG with transparent background

Proper lighting improves pose detection

This is a 2D image-based try-on, not a 3D simulation

Screenshots
🔹 Original Input Image

Front-facing person image uploaded by the user.

<img width="1024" height="1536" alt="user2" src="https://github.com/user-attachments/assets/309d08b8-1b2e-40fe-8534-42439bd6b2a0" />


🔹 Virtual Try-On Output

Shirt and pants automatically aligned based on body pose (neck, hips, waist, legs).

![11](https://github.com/user-attachments/assets/49298b0c-63ca-4005-9a98-070bc6e9f3c3)
![12](https://github.com/user-attachments/assets/9af8fa50-3327-45ee-a692-7a155ed12cb0)
![13](https://github.com/user-attachments/assets/6ddf056d-d4aa-4808-aa84-7e913ee312b9)
![14](https://github.com/user-attachments/assets/1405ab21-ad96-43af-9955-21b3d48fd9d4)
<img width="599" height="414" alt="Screenshot 2026-02-05 232636" src="https://github.com/user-attachments/assets/6d5b5194-d823-4b17-89e2-1797b0bfd29f" />







📌 Future Improvements

Cloth warping (VITON / CP-VTON)

Multiple clothing categories

Manual adjustment sliders

Better body segmentation

Realistic cloth deformation

👩‍💻 Author

Rabia Sarwar
GitHub: rabiasarwar726-maker
