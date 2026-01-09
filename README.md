🌟 Intelligent Image Enhancement System
Dark, Normal & Overexposed Image Correction using Deep Learning
📌 Project Overview

This project is an Intelligent Image Enhancement Web Application that automatically improves image quality based on its lighting condition.

The system:

Enhances low-light images using Zero-DCE deep learning model

Corrects overexposed (too bright) images using a dedicated overexposure correction model

Applies adaptive image processing filters for normal images

Automatically selects the best enhancement pipeline without user input

The application is built using Flask, TensorFlow, and OpenCV.

🎯 Key Features

🔍 Automatic image quality analysis (brightness & contrast)

🌑 Low-light enhancement using Zero-DCE

☀️ Overexposure correction for very bright images

⚙️ Adaptive CLAHE & contrast enhancement

🎨 Color balance correction

🔇 Noise reduction

🌐 Web-based interface using Flask

📂 Upload and view original & enhanced images

🧠 Enhancement Pipeline

The system follows three different pipelines based on image brightness:

🔻 Case 1: Low-Light Image
Upload → Preprocessing → ZeroDCE → CLAHE → Contrast → Color Balance → Denoising → Output

🔻 Case 2: Normal Image
Upload → Preprocessing → CLAHE → Contrast → Denoising → Output

🔻 Case 3: Overexposed Image
Upload → Preprocessing → Overexposure Correction → CLAHE → Contrast → Denoising → Output

🔄 Preprocessing Steps

Before enhancement, the following preprocessing steps are applied:

Image loading using OpenCV

BGR → RGB color conversion

Brightness and contrast calculation

Decision logic to select enhancement path

Model-specific resizing and normalization

🏗️ Technologies Used

Python

Flask – Web framework

TensorFlow / Keras – Deep learning

OpenCV – Image processing

NumPy – Numerical operations

HTML / CSS – Frontend

📁 Project Structure
├── static/
│   ├── uploads/        # Uploaded images
│   └── results/        # Enhanced images
├── templates/
│   └── index.html      # Web interface
├── zero_dce_model_weights.h5
├── app.py              # Main Flask application
├── README.md

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Install Required Libraries
pip install tensorflow flask opencv-python numpy

3️⃣ Run the Flask App
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/


📌 Applications

Low-light photography enhancement

Surveillance image improvement

Mobile camera post-processing

Medical and satellite image preprocessing

Image enhancement for computer vision tasks

🚀 Future Enhancements

Real-time video enhancement

Mobile app integration

GPU acceleration

User-controlled enhancement intensity

Support for batch image processing

👨‍💻 Author

Udaykumar G
B.Tech – Computer Science Engineering (AI & ML)

📜 License

This project is for educational and research purposes.
