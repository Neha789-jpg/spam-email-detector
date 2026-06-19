# 📧 Email Spam Detector

## Overview

This project is a Machine Learning-based Email/SMS Spam Detection application built using Python, TF-IDF Vectorization, and Logistic Regression.

The application classifies a message as either:

* ✅ Not Spam (Ham)
* 🚨 Spam

A simple Streamlit web interface allows users to enter a message and receive an instant prediction.

---

## Features

* Text preprocessing using TF-IDF Vectorization
* Spam classification using Logistic Regression
* Interactive Streamlit web application
* Real-time prediction
* Model serialization using Pickle

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Streamlit

---

## Dataset

The model was trained using the SMS Spam Collection Dataset, which contains labeled SMS messages classified as spam or ham.

---

## Project Structure

```text
├── app.py
├── spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Example

Input:

```text
Congratulations! You have won a FREE iPhone. Claim now!
```

Output:

```text
Spam
```

---

## Machine Learning Workflow

1. Load dataset
2. Convert labels to numerical values
3. Apply TF-IDF Vectorization
4. Split dataset into training and testing sets
5. Train Logistic Regression model
6. Evaluate model performance
7. Save model and vectorizer using Pickle
8. Deploy using Streamlit

---

## Results

* Model: Logistic Regression
* Feature Extraction: TF-IDF Vectorization
* Accuracy: Approximately 96%

---

## Future Improvements

* Phishing email detection
* Advanced text preprocessing
* Deep learning-based classification
* Email attachment analysis
* Multi-language spam detection

---

## Author

Neha Khundiya

Second-Year B.Tech CSE Student | Machine Learning Enthusiast
