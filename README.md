<div align="center">

# 💪 Flex.i

### *Elevate Your Fitness, Anytime Anywhere*

> **Datathon 2024** · Team: Untitled

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 📌 Problem Statement

Traditional fitness tracking methods often rely on **manual input**, which can be imprecise and lead to suboptimal results, negatively impacting the user's overall fitness journey.

**Our aim** is to explore and develop a state-of-the-art **pose estimation system** using Computer Vision and Machine Learning to be used in the field of fitness. The web application also assists in maintaining dietary planning, exercising, and calculating calories.

---

## 🧠 Overview

```
┌─────────────────────────┐     ┌──────────────────────────────┐     ┌───────────────────────────────┐
│          1              │     │             2                │     │              3                │
│                         │     │                              │     │                               │
│  Flex.i helps you to    │────▶│  Our web tool uses OpenCV,   │────▶│  Our model is capable of      │
│  track your own workout │     │  LSTM and MobileNet model    │     │  pose detection, calories     │
│  and diet plan remotely │     │  trained over a large data   │     │  identification, personalised │
│  without depending on   │     │  of images and videos.       │     │  recommendation system.       │
│  a professional.        │     │                              │     │                               │
└─────────────────────────┘     └──────────────────────────────┘     └───────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏋️ **Exercise Pose Detection** | Real-time detection and analysis of workout poses via webcam |
| 🔢 **Repetition Counter** | Automatic counting of exercise reps with precise pose tracking |
| 🥗 **Food Classification** | Upload a food image and instantly identify the food item |
| 🔥 **Nutrients / Calories Intake** | Get calorie estimates per 100g for classified food |
| 🤖 **Recommendation System** | Personalized workout and diet recommendations |
| 🎙️ **Voice Assistant** | Hands-free navigation and interaction within the app |

---

## 🤖 ML Models Used

### 1. 🥦 Food Classification & Calories Intake Model

We used **MobileNetV2** — a lightweight convolutional neural network designed for mobile applications, balancing accuracy with computational efficiency.

**How it works:**
- Uses **depthwise separable convolutions**, inverted residuals, and linear bottleneck layers
- Trained on a dataset of food images labeled with **36 classes**
- Accurately classifies various food items and returns calorie data per 100g

**Use cases:** Food recognition, dietary tracking

```
📸 Upload food image
         │
         ▼
  ┌─────────────┐
  │ MobileNetV2 │  ──── 36-class food classifier
  └─────────────┘
         │
         ▼
  Category: Fruit
  Predicted: Kiwi
  61 calories / 100g
```

---

### 2. 📊 Calories Intake Prediction Model (BMR)

**BMR (Basal Metabolic Rate)** is the number of calories the body burns at rest to perform basic life-sustaining functions.

We incorporated this relationship into a **Machine Learning model** that predicts the calories required for basic body functioning using the following inputs:

| Input | Description |
|---|---|
| `Gender` | Male / Female |
| `Age` | User's age |
| `Weight` | In kilograms |
| `Height` | In centimeters |

> **Example output:** *Predicted Calorie Intake: 1907.99 kcal*

---

### 3. 🏃 Customized Pose Detection Fitness Model

An advanced fitness model utilizing **OpenCV** and **LSTM networks**.

**Capabilities:**
- ✅ Accurately identifies exercises (bicep curls, push-ups, squats, and more)
- ✅ Built-in **repetition counter** per exercise
- ✅ Ensures correct form via **precise pose keypoint analysis**
- ✅ Provides comprehensive feedback on workout performance
- ✅ Promotes safe and effective exercise practices

**Architecture:**

```
📹 Live Video / Webcam Feed
         │
         ▼
  ┌─────────────┐
  │   OpenCV    │  ──── Frame extraction & preprocessing
  └─────────────┘
         │
         ▼
  ┌─────────────┐
  │    LSTM     │  ──── Temporal sequence modeling
  └─────────────┘
         │
         ▼
  Exercise Identified + Rep Count + Form Feedback
```

**Exercises supported:** `bicep_curl` · `push_ups` · `squat`

---

## 🛠️ Tech Stack

### Machine Learning & Data Science
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat-square&logo=OpenCV&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)

### Frontend
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)

### Backend
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)

---

## 🌐 Web Application

The Flex.i web app is structured around four key sections:

| Section | Description |
|---|---|
| 🏠 **Home** | Landing page with app overview and voice assistant access |
| 🥗 **Diet** | Diet Planner — search recipes, view calorie counts, plan meals |
| 🏋️ **Exercise** | Pose detection & repetition counter via live camera feed |
| 🔥 **Calorie** | Food image classifier + BMR-based calorie intake predictor |

---

## 💡 Unique Selling Point

| Competitors | Flex.i |
|---|---|
| Focus on manual data entry | **AI-powered automatic tracking** |
| Requires in-person trainer | **100% remote access** |
| Generic recommendations | **Personalized recommendation system** |
| Separate tools for diet & exercise | **All-in-one platform** |
| Time-consuming workflows | **Efficient, real-time results** |

> *"Competitors focus more on manual training whereas our idea integrates AI with new age technology and gives more accurate results."*

---

## 📁 Project Structure

```
flex-i/
├── backend/
│   ├── app.py                  # Flask application entry point
│   ├── models/
│   │   ├── pose_model/         # OpenCV + LSTM pose detection model
│   │   ├── food_classifier/    # MobileNetV2 food classification model
│   │   └── bmr_model/          # BMR calorie intake prediction model
│   └── utils/
│       └── voice_assistant.py  # Voice assistant integration
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       │   ├── Home.jsx
│       │   ├── Diet.jsx
│       │   ├── Exercise.jsx
│       │   └── Calorie.jsx
│       └── App.jsx
├── notebooks/
│   ├── food_classification.ipynb
│   ├── pose_detection.ipynb
│   └── bmr_prediction.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam (for pose detection)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/flex-i.git
cd flex-i

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install

# Start the Flask backend
cd ../backend
python app.py

# In a new terminal, start the React frontend
cd frontend
npm start
```

### Running the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
tensorflow>=2.10
keras
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
flask
flask-cors
mediapipe
```

---

<div align="center">

**Flex.i** · Datathon 2024 · Team Untitled

*Elevate Your Fitness, Anytime Anywhere* 💚

</div>
