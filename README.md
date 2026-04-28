# IT Helpdesk Ticket Classifier

A machine learning model that reads IT support ticket text and automatically classifies it into one of eight categories. Built with Python, Scikit-learn, and TF-IDF vectorization. Includes a fully functional Streamlit web application for single and bulk ticket classification.

---

## Overview

In a real IT department, support tickets arrive continuously and someone has to read each one to decide which team should handle it. This project automates that process using natural language processing and a Logistic Regression classifier.

The model reads the ticket text, identifies key patterns, and predicts the correct category with 84.53% accuracy. The project is deployed as a local web app using Streamlit.

---

## Categories

The model classifies tickets into eight categories:

- Hardware
- HR Support
- Access
- Miscellaneous
- Storage
- Purchase
- Internal Project
- Administrative rights

---

## Project Structure

```
IT_Helpdesk_Classifier/
├── model/
│   ├── model.pkl         # Trained Logistic Regression classifier
│   └── tfidf.pkl         # Fitted TF-IDF vectorizer
├── app.py                # Streamlit web application
├── requirements.txt      # Required Python libraries
└── README.md             # Project documentation
```

---

## Dataset

- Source: [Kaggle — IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
- File: `all_tickets_processed_improved_v3.csv`
- Total records: 47,837
- Columns: `Document` (ticket text), `Topic_group` (category label)
- Missing values: None

| Category              | Ticket Count |
|-----------------------|--------------|
| Hardware              | 13,617       |
| HR Support            | 10,915       |
| Access                | 7,125        |
| Miscellaneous         | 7,060        |
| Storage               | 2,777        |
| Purchase              | 2,464        |
| Internal Project      | 2,119        |
| Administrative rights | 1,760        |

---

## Tools and Libraries

| Library                | Purpose                                          |
|------------------------|--------------------------------------------------|
| Python 3               | Programming language                             |
| Pandas                 | Data loading and exploration                     |
| Scikit-learn           | TF-IDF vectorization, model training, evaluation |
| Matplotlib             | Data visualization                               |
| Seaborn                | Confusion matrix heatmap                         |
| Streamlit              | Web application framework                        |
| streamlit-option-menu  | Horizontal tab navigation in the app             |
| Pickle                 | Saving and loading model files                   |
| Google Colab           | Model training environment                       |
| PyCharm                | Local development environment                    |

---

## How It Works

### Step 1 — Load Data
The dataset is downloaded directly from Kaggle using the Kaggle API and loaded into a Pandas DataFrame.

### Step 2 — Clean Text
Each ticket goes through a cleaning function that converts text to lowercase, removes special characters and numbers, and strips extra whitespace.

### Step 3 — TF-IDF Vectorization
The cleaned text is converted into numerical features using TF-IDF. The top 10,000 words are selected as features and standard English stop words are removed.

### Step 4 — Train and Test Split

| Split        | Percentage | Samples |
|--------------|------------|---------|
| Training set | 80%        | 38,269  |
| Testing set  | 20%        | 9,568   |

### Step 5 — Train the Model
A Logistic Regression classifier is trained on the TF-IDF features. Training completed in 7.39 seconds on Google Colab.

### Step 6 — Save Model
The trained model and vectorizer are saved as pickle files for use in the Streamlit app.

```python
import pickle

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
```

---

## Results

Overall Accuracy: 84.53%

| Category              | Precision | Recall | F1 Score | Test Samples |
|-----------------------|-----------|--------|----------|--------------|
| Access                | 0.92      | 0.87   | 0.89     | 1,455        |
| Administrative rights | 0.87      | 0.68   | 0.76     | 342          |
| HR Support            | 0.85      | 0.83   | 0.84     | 2,107        |
| Hardware              | 0.79      | 0.88   | 0.83     | 2,760        |
| Internal Project      | 0.91      | 0.80   | 0.86     | 451          |
| Miscellaneous         | 0.80      | 0.82   | 0.81     | 1,400        |
| Purchase              | 0.97      | 0.88   | 0.92     | 497          |
| Storage               | 0.93      | 0.84   | 0.88     | 556          |
| Weighted Average      | 0.85      | 0.85   | 0.85     | 9,568        |

---

## Streamlit Web Application

The project includes a Streamlit app with three tabs:

| Tab              | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Single Ticket    | Enter a ticket description and get the predicted category with confidence score |
| CSV Upload       | Upload a CSV with a Document column to classify tickets in bulk with downloadable results |
| About the Model  | View model details, training information, and per-category performance metrics |

---

## How to Run Locally

1. Clone or download this repository
2. Open the folder in PyCharm
3. Create a virtual environment:
```bash
python -m venv venv
```
4. Activate the virtual environment:

Windows:
```bash
venv\Scripts\activate
```
Mac or Linux:
```bash
source venv/bin/activate
```
5. Install required libraries:
```bash
pip install -r requirements.txt
```
6. Run the app:
```bash
streamlit run app.py
```
7. The app will open in your browser at `https://ticketsense.streamlit.app/

---

## Custom Prediction Example

| Ticket Text                                                      | Predicted Category    |
|------------------------------------------------------------------|-----------------------|
| My laptop screen is broken and I cannot turn it on              | Hardware              |
| I need access to the HR portal to submit my leave request       | HR Support            |
| Please give me admin rights to install software on my computer  | Administrative rights |
| My external hard drive is not showing up on the network storage | Storage               |
| I want to purchase a new keyboard and mouse for my workstation  | Hardware              |

---

## Possible Improvements

- Apply oversampling (SMOTE) to handle class imbalance in smaller categories
- Try more advanced models such as Random Forest, SVM, or BERT
- Add a confidence threshold to flag uncertain predictions for manual review

---
