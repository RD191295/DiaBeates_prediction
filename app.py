import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pickle
import time
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from pymongo.mongo_client import MongoClient  # type: ignore
from pymongo.server_api import ServerApi  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# Retrieve credentials securely from Streamlit Secrets
mongo_uri = st.secrets["mongodb"]["uri"]
db_name = st.secrets["mongodb"]["database"]
collection_name = st.secrets["mongodb"]["collection"]

# Connect to MongoDB
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
database = client[db_name]
collection = database[collection_name]



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
    
def plot_input_data(user_data):
    
    """
    Plots the input user data as a bar chart with theme-aware styling.

    This function filters numeric data from the provided user data dictionary
    and creates a bar plot using Matplotlib. The plot's appearance is adjusted
    based on the current Streamlit theme, ensuring optimal visibility in both
    light and dark modes. The plot is displayed directly in a Streamlit app.

    Args:
        user_data (dict): A dictionary containing user input data with keys as
                          parameter names and values as numeric data.

    Returns:
        None: This function displays the plot using Streamlit and does not
              return any value.
    """
    # Filter numeric columns
    user_data = {k: v for k, v in user_data.items() if isinstance(v, (int, float))}

    # Detect Streamlit theme
    st_theme = st.get_option("theme.base")  # 'light' or 'dark'
    
    # Adjust colors based on theme
    if st_theme == "dark":
        sns.set_style("darkgrid")  # Dark theme grid
        text_color = "white"
        grid_color = "gray"
        bar_color = "cyan"  # Bright color for dark mode
    else:
        sns.set_style("whitegrid")  # Light theme grid
        text_color = "black"
        grid_color = "lightgray"
        bar_color = "royalblue"  # Darker color for light mode

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")  # Transparent background

    # Create bar plot with a single color for visibility
    bars = ax.bar(user_data.keys(), user_data.values(), color=bar_color, edgecolor="black", linewidth=1.2)
    
    # Set labels and title with theme-aware colors
    plt.xticks(rotation=45, fontsize=12, ha="right", color="white")
    plt.xlabel("Parameters", fontsize=14, fontweight="bold", color="white")
    plt.yticks(color="white")  # Set y-axis values color
    plt.ylabel("Values", fontsize=14, fontweight="bold", color="white")
    plt.title("Patient Input Data & Predictions", fontsize=16, fontweight="bold", color="white")

    # Add grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=grid_color)

    # Set background color to match theme
    ax.set_facecolor("none")  # Transparent, blends with Streamlit background

    # Add values on top of bars with contrasting text color
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 5), 
                    textcoords="offset points", 
                    ha="center", 
                    fontsize=12, 
                    fontweight="bold",
                    color="white")

    # Display the plot in Streamlit
    st.pyplot(fig)

# Main Streamlit App
def main():
    """Streamlit app for diabetes prediction."""
    st.set_page_config(page_title="🔬 Diabetes Predictor", page_icon="🩺", layout="wide")

    st.title("🚀 **AI-Based Diabetes Progression Prediction**")
    st.markdown("🔬 **Using Machine Learning to Estimate Diabetes Progression Over Time**")

    # 🎨 Custom Background Styling
    st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ffefba, #ffffff);
        }
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    # Sidebar Layout
    st.sidebar.header("⚙️ **Configuration**")
    st.sidebar.write("Select the prediction model.")

    model_choice = st.sidebar.radio("Choose Model", ["🤖 Ridge Regression", "🧮 Lasso Regression"])

    st.sidebar.markdown("---")  # Add a divider

    st.sidebar.header("📋 **Patient Data Input**")
    st.sidebar.write("Fill in the details below to predict diabetes progression.")
    
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
        with st.spinner("🕒 Processing your input... Please wait"):
            time.sleep(2)  # Simulate a loading delay
         
            # Map Model Selection
            model_name = "Ridge_model.pkl" if model_choice == "🤖 Ridge Regression" else "Lasso_model.pkl"
            model_Id = "Ridge" if model_choice == "🤖 Ridge Regression" else "Lasso"

            # Make Prediction
            prediction = predict_data(user_data, model_name)
             
            # Store user data in MongoDB
            user_data["quantitative measure of disease progression"] = float(prediction[0])
            user_data["model_name"] = model_Id
            
            # Convert NumPy types to Python native types
            document = {key: int(value) if isinstance(value, np.integer) else   
                     float(value) if isinstance(value, np.floating) else value
                for key, value in user_data.items()}
            
            collection.insert_one(document)
        
        # Display Results
        st.markdown(f"## 🎯 Prediction Result")
        st.success(f"📊 **Estimated Disease Progression Score: {prediction[0]:.2f}**")

        # Show Data Visualization
        plot_input_data(user_data)
        
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
        
     # 👨‍💻 Footer - Created By
    st.markdown(
        "<br><hr><center>🚀 Created by **Your Name** | Made with ❤️ using Streamlit</center><hr>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
