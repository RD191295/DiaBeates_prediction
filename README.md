# 🔬 AI-Based Diabetes Progression Prediction

## 📌 Overview
This is a **machine learning-powered web application** built with **Streamlit** that predicts diabetes progression over time based on multiple clinical factors such as age, BMI, blood pressure, serum cholesterol, and more. The app uses **Ridge Regression** and **Lasso Regression** models for prediction and securely stores user input data in **MongoDB**.

## 🚀 Features
- **User-Friendly Interface:** Clean and interactive UI with **light/dark mode support**.
- **Model Selection:** Choose between **Ridge Regression** and **Lasso Regression**.
- **Real-time Predictions:** Predict diabetes progression instantly based on input parameters.
- **Data Visualization:** Displays user input data as a **professional bar chart**.
- **Secure Data Storage:** Saves user predictions in **MongoDB** for analysis.
- **Themed UI:** Adaptive color scheme based on Streamlit's theme.

## 🏗️ Installation
### Prerequisites
Make sure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **MongoDB Atlas** (or a local MongoDB instance)

### Steps to Set Up
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate  # For Windows
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up Streamlit Secrets for MongoDB:**
   - Create a `.streamlit/secrets.toml` file:
   ```toml
   [mongodb]
   uri = "your_mongodb_connection_string"
   database = "your_database_name"
   collection = "your_collection_name"
   ```
5. **Run the app:**
   ```sh
   streamlit run app.py
   ```

## 🎯 How to Use
1. Open the app in your browser after running it.
2. Select a **prediction model** from the sidebar.
3. Enter the **patient details** (age, sex, BMI, blood pressure, etc.).
4. Click **Predict** to generate a **diabetes progression score**.
5. View **visualized data insights** and predictions.
6. The data is stored in MongoDB for future reference.

## 📊 Model Details
- **Ridge Regression**: Reduces overfitting by adding an L2 penalty.
- **Lasso Regression**: Feature selection by applying an L1 penalty.

## 📦 Technologies Used
- **Frontend & Backend:** Streamlit
- **Machine Learning Models:** Scikit-learn
- **Database:** MongoDB
- **Visualization:** Matplotlib, Seaborn

## 🛠️ Future Enhancements
- Integrate additional machine learning models.
- Provide explanations using **SHAP (SHapley Additive exPlanations)**.
- Deploy to **Streamlit Cloud** or **Heroku**.

## 📜 Disclaimer
🚨 *This is a machine learning-based prediction tool and should not be used as a substitute for professional medical advice. Consult a doctor for an accurate clinical assessment.*

## 👨‍💻 Author
Developed by **Raj Dalsaniya** with ❤️ using Streamlit.

---
✨ **If you like this project, give it a ⭐ on GitHub!** ✨
