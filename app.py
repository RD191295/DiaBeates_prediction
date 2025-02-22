import streamlit as st # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pickle
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from pymongo.mongo_client import MongoClient # type: ignore
from pymongo.server_api import ServerApi # type: ignore

#uri = "mongodb+srv://rd191295:Rd191295@cluster0.gteaj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
#client = MongoClient(uri, server_api=ServerApi('1'))
#database = client['Diabetes_prediction']
#collection = database['Diabetes_data']  

def load_model(model_name):
    """Loads the saved model and its dependencies from a pickle file."""
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler


def processing_input_data(data, scaler):
    """Processes the input data to prepare it for the model."""
    data = pd.DataFrame([data])
    data["SEX"] = data["SEX"].map({"Male": 1, "Female": 2})
    data_transformed = scaler.transform(data)
    print(data_transformed)
    return data_transformed

def predict_data(data, model_name):
    """Predicts the output based on the input data."""
    model, scaler= load_model(model_name)
    processed_data = processing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction


def main():
    """
    The main function contains the Streamlit app logic. It sets the page configuration, 
    defines the UI layout and elements, processes user input data, makes a prediction using the
    loaded model, and displays the results. Additionally, it stores the user data in a MongoDB
    collection.

    """
    st.set_page_config(page_title="Quanitfied Diabetes Prediction", layout="wide")

    st.title("Quanitfied Diabetes Prediction")
    st.markdown("""
        <style>
            .main-title {
                font-size: 24px;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Layout
    model_name= st.sidebar.selectbox("Select Model", ["Ridge Regression", "Lasso Regression"])
    st.sidebar.header("Input Parameters")
    age = st.sidebar.slider("Age of Patient", 18, 80, 19)
    sex = st.sidebar.selectbox("Sex of Patient", ["Male", "Female"])
    bmi = st.sidebar.slider("BMI of Patient", 18.0, 43.0,19.2)
    bp = st.sidebar.slider("Blood Pressure of Patients", 60, 180, 120)
    s1 = st.sidebar.slider("Total Serum Cholesterol", 90, 200, 400)
    s2 = st.sidebar.slider("Low-Density Lipoproteins", 50, 100, 250)
    s3 = st.sidebar.slider("High-Density Lipoproteins", 20, 40, 100)
    s4 = st.sidebar.slider("Total Cholesterol / HDL", 1.5, 0.1, 10.0)
    s5 = st.sidebar.slider("possibly log of serum triglycerides level", 3.0, 5.2, 6.5)
    s6 = st.sidebar.slider("blood sugar level", 50, 99, 600)
    
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

    #print (user_data)

    if st.sidebar.button("Predict"):
        # Make Prediction
        if(model_name == "Ridge Regression"):
            model_name = "Ridge_model.pkl"
        elif(model_name == "Lasso Regression"):
            model_name = "Lasso_model.pkl"
        
        prediction = predict_data(user_data,model_name)
        
        #print(type(float(prediction[0])))
        # Store user data in MongoDB
        user_data["quantitative measure of disease progression"] = float(prediction[0])
        user_data["model_name"] = model_name

        # Convert NumPy types to Python native types
        #document = {key: int(value) if isinstance(value, np.integer) else   
        #         float(value) if isinstance(value, np.floating) else value
        #    for key, value in user_data.items()}
        
        #collection.insert_one(document)

         # Display Prediction Results
        st.markdown(f"### Prediction Result")
        st.markdown(f"""
            **Predicted Quantitive Measure of Disease Progression after one year is**: 
            **{prediction[0]:.2f}**
        """, unsafe_allow_html=True)
        
        # Additional info (can be customized further)
        st.markdown("### Explanation of Prediction")
        st.write("The prediction is based on the input factors such as age, sex, BMI, blood pressure, serum cholesterol, HDL, and triglycerides. The result indicates quantitive measurement of disease progression after 1 year")
        
    # Add footer or additional elements (optional)
    st.markdown("""
        ---
        *Note: The prediction is based on the data model and may not fully capture all real-world factors. Please use it as a guideline.*
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()