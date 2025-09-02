
# Gender Categorization

Gender_categorization is a machine learning model that predicts gender based on English names. It uses a character-level n-gram approach with a logistic regression classifier to provide fast, interpretable, and highly accurate predictions — achieving **96%+ accuracy** on both validation and test sets.

## 🧠 Model Overview

- **Type:** Classic ML (Logistic Regression)
- **Input:** English name (flexible: single or full name)
- **Vectorization:** Character-level n-grams (2–3 chars)
- **Framework:** scikit-learn
- **Training Set:** 117,806 names (out of 147,257)
- **Validation/Test Accuracy:** ~96.6%

---

## 📁 Project Structure

```
Gender_categorization/
├── data/
│   └── Gender_prediction.csv
├── models/
│   ├── logistic_model.joblib
│   └── vectorizer.joblib
├── train.py
├── interactive.py
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Predict gender from a name using script
Run interactive inference with:
```bash
python interactive.py
```

---

## 📊 Training & Evaluation

- The model is trained using [`train.py`](train.py).
- Metrics (accuracy, precision, recall, F1-score) are logged and summarized after training.
- Model and vectorizer are saved in the `models/` directory for later use.

---

## 📝 Data

- Source: [`data/Gender_prediction.csv`](data/Gender_prediction.csv)
- Columns: `name`, `gender`
- NaN values in these columns are dropped before training.

---

## 🛠 Dependencies

See [`requirements.txt`](requirements.txt) for all required packages. Key dependencies:
- scikit-learn
- pandas
- numpy
- tabulate
- joblib

---

## 📄 License

This project is maintained by [Ian Gitonga] and is open for research and educational use.
