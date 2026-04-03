# Smart Agriculture Optimization: Crop Recommendation System 🌱

## Overview
This project is an End-to-End Machine Learning Pipeline designed to empower farmers and agricultural stakeholders by predicting the most optimal crop to plant based on current soil and weather conditions. 

The system leverages a **Random Forest Classifier** trained on historical data of 22 different crops, analyzing parameters such as Nitrogen, Phosphorus, Potassium (NPK), pH, Temperature, Humidity, and Rainfall to provide highly accurate business intelligence.

## 🚀 Features
- **High Accuracy Model:** Achieved 99.1% predictive accuracy.
- **Explainable AI (XAI):** Insights show Rainfall and Humidity control >43% of the predictive logic, proving the AI learned natural botanical behavior.
- **Interactive Top 5 Alternatives:** Uses `predict_proba` to deliver highly flexible crop options with percentages instead of strict 1-choice outputs.
- **BI Dashboard:** Fully tailored Streamlit Right-To-Left (RTL) Arabic User Interface handling prediction inference and data storytelling.

## 📁 Project Structure
- `app.py`: The Main Streamlit Dashboard (Product inference & Presentation mode).
- `linkedin_dashboard.py`: A secondary focused dashboard specifically built for presentation.
- `model_training.ipynb` / `clean_notebook.py`: The full exploratory data analysis (EDA), scaling, and algorithm training phase.
- `crop_model.pkl` & `crop_scaler.pkl`: Serialized joblib weights for production inference.
- `linkedin_post_infographic.png`: A quick-look infographic highlighting the model parameters and success metrics.

## 🛠️ Tech Stack
- **Python** (Pandas, Numpy)
- **Machine Learning:** Scikit-Learn
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Deployment:** Streamlit
- **Serialization:** Joblib

## ⚙️ How to Run Locally
1. Clone the repository.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
