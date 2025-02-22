import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from pymongo.mongo_client import MongoClient  # type: ignore
from pymongo.server_api import ServerApi  # type: ignore

# Retrieve credentials securely from Streamlit Secrets
mongo_uri = uri = "mongodb+srv://bitcoinee12:Rd191295@diacluster0.lw673.mongodb.net/?retryWrites=true&w=majority"


client = MongoClient(uri, server_api=ServerApi('1'))
database = client['Diabetes_prediction']
collection = database['Diabetes_data']  

# Function to load the model
def load_model(model_name):
    """Loads the saved model and scaler from a pickle file."""
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# Function to process input data
def processing_input_data(data, scaler):
    """Processes input data before prediction."""
    data = pd.DataFrame([data])
    data["SEX"] = data["SEX"].map({"Male": 1, "Female": 2})
    data_transformed = scaler.transform(data)
    return data_transformed

# Function for prediction
def predict_data(data, model_name):
    """Predicts the output based on input data."""
    model, scaler = load_model(model_name)
    processed_data = processing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction

# Main Streamlit App
def main():
    """Streamlit app for diabetes prediction."""
    st.set_page_config(page_title="🔬 Diabetes Progression Prediction", layout="wide")

    st.title("🔍 **Quantified Diabetes Progression Predictor**")
    st.markdown("🔬 **Using Machine Learning to Estimate Diabetes Progression Over Time**")

    # Sidebar Layout
    st.sidebar.header("⚙️ **Model Selection**")
    model_choice = st.sidebar.radio("Choose Model", ["🤖 Ridge Regression", "🧮 Lasso Regression"])

    st.sidebar.header("📋 **Input Parameters**")
    age = st.sidebar.slider("🎂 Age of Patient", 18, 80, 25)
    sex = st.sidebar.selectbox("⚤ Sex of Patient", ["Male", "Female"])
    bmi = st.sidebar.slider("⚖️ BMI of Patient", 18.0, 43.0, 25.0)
    bp = st.sidebar.slider("💉 Blood Pressure", 60, 180, 120)
    s1 = st.sidebar.slider("🩸 Total Serum Cholesterol", 90, 400, 200)
    s2 = st.sidebar.slider("🧪 Low-Density Lipoproteins (LDL)", 50, 250, 100)
    s3 = st.sidebar.slider("💊 High-Density Lipoproteins (HDL)", 20, 100, 50)
    s4 = st.sidebar.slider("🔗 Total Cholesterol / HDL Ratio", 1.5, 10.0, 4.5)
    s5 = st.sidebar.slider("🩺 Log of Serum Triglycerides", 3.0, 6.5, 5.2)
    s6 = st.sidebar.slider("🩸 Blood Sugar Level", 50, 600, 99)

    user_data = {
        "AGE": age,
        "SEX": sex,
        "BMI": bmi,
        "BP": bp,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        "S5": s5,
        "S6": s6
    }

    if st.sidebar.button("🚀 Predict"):
        # Map Model Selection
        model_name = "Ridge_model.pkl" if model_choice == "🤖 Ridge Regression" else "Lasso_model.pkl"
        
        # Make Prediction
        prediction = predict_data(user_data, model_name)
         
        # Store user data in MongoDB
        user_data["quantitative measure of disease progression"] = float(prediction[0])
        user_data["model_name"] = model_name
        # Convert NumPy types to Python native types
        document = {key: int(value) if isinstance(value, np.integer) else   
                 float(value) if isinstance(value, np.floating) else value
            for key, value in user_data.items()}
        
        collection.insert_one(document)
        
        # Display Results
        st.markdown(f"## 🎯 Prediction Result")
        st.success(f"📊 **Estimated Disease Progression Score: {predicted_value:.2f}**")

        # Explanation Section
        st.markdown("## 📌 Explanation of Prediction")
        st.info(
            "The prediction is based on multiple clinical factors such as age, sex, BMI, "
            "blood pressure, serum cholesterol, LDL, HDL, and triglycerides. The result indicates "
            "a **quantitative measure of disease progression** after one year."
        )

        # Disclaimer
        st.markdown(
            "---\n"
            "⚠️ *Note: This is a machine learning-based prediction and should not be considered a definitive diagnosis. "
            "Consult a medical professional for accurate clinical assessment.*",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
