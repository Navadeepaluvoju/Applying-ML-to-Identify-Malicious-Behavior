### Requirements: URL Phishing Attack Using Machine Learning

# 🧠 NLP Prediction Web App

This project is a **Flask-based web application** that enables users to input data and receive real-time predictions using **Natural Language Processing (NLP)** and **Logistic Regression**. The app includes a login system, home dashboard, and a detection page for analyzing user input—perfect for text classification tasks like sentiment analysis or spam detection.

## 🚀 Features

- User Login & Session Management
- Clean UI for text input
- Text preprocessing using NLP techniques
- Prediction using a Logistic Regression model trained on sample data
- Result display with redirect and error handling

## 🛠️ Tech Stack

- **Python**, **Flask**
- **Pandas**, **NumPy**
- **Scikit-learn** (for Logistic Regression model)
- **HTML/CSS**
- Local file-based model loading (e.g., `.pkl`)

## 📂 Folder Structure

```
Flask/
├── app.py
├── static/
├── templates/
├── Images/
│   ├── Login.png
│   └── Home.png
│   └── detection_page.png
├── model/
│   └── model.pkl
├── requirements.txt
```

## ✅ Setup Instructions

1. **Clone this repo**:
   ```bash
   git clone https://github.com/yourusername/nlp-flask-app.git
   cd Flask
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

4. **Run the app**:
   ```bash
   set FLASK_APP=app.py       # On Windows
   flask run
   ```

   Or simply:
   ```bash
   python app.py
   ```

5. **Visit your browser**:
   ```
   http://127.0.0.1:5000
   ```

## 📊 Model Details

- The text classification is powered by **Logistic Regression**.
- Input data is preprocessed using basic NLP methods (e.g., lowercasing, removing punctuation, tokenization).
- Trained model is stored in `/model/model.pkl`.
