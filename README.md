<div align="center">

# 🚦 Traffic Volume Predictor

### AI-powered Metro Interstate Traffic Volume Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://traffic-volume-predictor-001.streamlit.app/)

</div>

---

## 📌 Overview

**Traffic Volume Predictor** is an end-to-end machine learning application that predicts metro interstate traffic volume in real time. It combines **historical traffic data** with **weather and temporal features** to deliver accurate vehicle count estimates per hour — all through a clean, interactive Streamlit web app.

Built using a **Random Forest Regression** model trained on the Metro Interstate Traffic Volume dataset, this project covers the complete ML pipeline: from data exploration and feature engineering to model persistence and deployment.

---

## ✨ Features

- 🌤️ **Weather-aware predictions** — temperature, rain, snow, cloud coverage
- 🕐 **Time-based inputs** — hour, day, month, day of week, weekend flag
- 🏖️ **Holiday recognition** — handles major US holidays (Christmas, Thanksgiving, New Year's)
- 🌫️ **Weather condition support** — Clear, Clouds, Rain, Snow, Mist, Fog
- 📦 **Persisted model artifacts** — pre-trained model, scaler, and feature columns saved as `.pkl`
- ⚡ **Real-time inference** — instant predictions via a user-friendly slider/input UI
- ☁️ **Deployed on Streamlit Cloud** — accessible anywhere, no setup required

---

## 🖥️ Live Demo

Try the app live at:

**🔗 [https://traffic-volume-predictor-001.streamlit.app/](https://traffic-volume-predictor-001.streamlit.app/)**

---

## 🧠 How It Works

```
User Inputs (weather + time)
        │
        ▼
  Feature Engineering
  (One-Hot Encoding for holidays & weather)
        │
        ▼
  StandardScaler (pre-fitted)
        │
        ▼
  Random Forest Regressor
        │
        ▼
  Predicted Traffic Volume (vehicles/hour)
```

The model was trained on the [UCI Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume), which includes hourly traffic data from Minnesota DOT along with corresponding weather readings.

---

## 🗂️ Project Structure

```
Traffic-Volume-Predictor/
│
├── app.py                                  # Main Streamlit application
├── ap.py                                   # Alternate/experimental app script
├── metro-interstate-traffic-volume.ipynb   # EDA, training & model evaluation notebook
├── feature_columns.pkl                     # Saved feature column order
├── scaler.pkl                              # Fitted StandardScaler
├── requirements.txt                        # Python dependencies
└── LICENSE                                 # MIT License
```

> **Note:** `traffic_volume_model.pkl` (the trained Random Forest model) is loaded at runtime by `app.py`. Make sure it is present in the root directory when running locally.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Asad-Aziz-001/Traffic-Volume-Predictor.git
cd Traffic-Volume-Predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🧩 Input Features

| Feature | Type | Description |
|---|---|---|
| `Temperature` | Float | Atmospheric temperature in Kelvin |
| `Rain (1h)` | Float | Rainfall in the last hour (mm) |
| `Snow (1h)` | Float | Snowfall in the last hour (mm) |
| `Cloud Coverage` | Integer | Cloud cover percentage (0–100%) |
| `Hour` | Integer | Hour of the day (0–23) |
| `Day` | Integer | Day of the month (1–31) |
| `Month` | Integer | Month of the year (1–12) |
| `Day of Week` | Integer | 0 = Monday, 6 = Sunday |
| `Is Weekend` | Binary | 1 if Saturday/Sunday, else 0 |
| `Holiday` | Categorical | None / Christmas / Thanksgiving / New Year's |
| `Weather Condition` | Categorical | Clear, Clouds, Rain, Snow, Mist, Fog |

---

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Feature Scaling | StandardScaler |
| Encoding | One-Hot Encoding (holiday & weather) |
| Persistence | `joblib` (`.pkl` files) |
| Dataset | Metro Interstate Traffic Volume (UCI) |
| Target Variable | `traffic_volume` (vehicles/hour) |

---

## 📦 Dependencies

```txt
streamlit
pandas
numpy
scikit-learn
joblib
```

Install all at once with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Deployment

This project is deployed on **Streamlit Community Cloud**. To deploy your own fork:

1. Push your code (including `.pkl` model files) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the main file path to `app.py`.
4. Click **Deploy** — your app will be live in minutes!

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Asad Aziz**

[![GitHub](https://img.shields.io/badge/GitHub-Asad--Aziz--001-181717?style=flat-square&logo=github)](https://github.com/Asad-Aziz-001)

[![LinkedIn](https://img.shields.io/badge/LINKEDIN-Asad--Aziz--ai-181717?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/asad-aziz-ai)

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

</div>
