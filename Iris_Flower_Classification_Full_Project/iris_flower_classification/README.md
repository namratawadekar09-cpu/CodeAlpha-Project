# Iris Flower Classification - Full Stack ML Project

This project uses:
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Machine Learning: Scikit-learn Logistic Regression
- Dataset: Built-in Iris dataset from Scikit-learn

## 1. Project structure

```text
iris_flower_classification/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   └── iris_model.joblib       # generated after training
├── backend/
│   └── __init__.py
└── frontend/
    ├── templates/
    │   └── index.html
    └── static/
        ├── style.css
        └── script.js
```

## 2. Run in VS Code

Open the project folder in VS Code terminal.

### Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt terminal or run:

```bash
venv\Scripts\activate.bat
```

### Install libraries

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

This creates the file:

```text
model/iris_model.joblib
```

### Start the Flask backend

```bash
python app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

## 3. How the application works

1. User enters sepal length, sepal width, petal length and petal width.
2. Frontend sends the values to the Flask API.
3. Backend loads the trained ML model.
4. Model predicts the Iris species.
5. Backend returns the prediction and probability.
6. Frontend displays the result.

## 4. API endpoint

### POST `/predict`

Example JSON:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "success": true,
  "prediction": "setosa",
  "confidence": 0.98
}
```
