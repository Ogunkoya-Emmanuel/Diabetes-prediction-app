# 🩺 Diabetes Prediction Web App

A machine learning web app that predicts whether a person is diabetic based on medical input data. Built with a Support Vector Machine (SVM) trained on the Pima Indians Diabetes Dataset and deployed using Streamlit.

---

## How It Works

1. The trained SVM model and StandardScaler are loaded from `.sav` files.
2. The user enters eight health parameters through the web form.
3. The input is standardized using the same scaler used during training.
4. The model returns a prediction: diabetic or not diabetic.

---

## Run Locally

```bash
git clone https://github.com/Ogunkoya-Emmanuel/Diabetes-prediction-app.git
cd Diabetes-prediction-app
pip install -r requirements.txt
streamlit run "Diabetes prediction web app.py"
```

Then open `http://localhost:8501` in your browser.

---

## File Structure

| File | Description |
|------|-------------|
| `Diabetes prediction web app.py` | Streamlit app (main entry point) |
| `Diabetes_Prediction.ipynb` | Notebook used to train and export the model |
| `trained_model.sav` | Saved SVM classifier |
| `scaler.sav` | Saved StandardScaler for input preprocessing |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

---

## Technologies

Python, Streamlit, Scikit-learn, Pandas, NumPy, Pickle

---

## Author

OGUNKOYA EMMANUEL OLUWAKEMI
 
📧 emmanuelscholarships777@gmail.com
🔗 [https://github.com/Ogunkoya-Emmanuel](https://github.com/Ogunkoya-Emmanuel)