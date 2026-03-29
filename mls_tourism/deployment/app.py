import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="v-vasanth2009/tourism-package-prediction-28032026", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a Customer purchasing Wellness Tourism package based on customer and interaction data.
Please enter the Customer and interaction data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (min)", min_value=0, max_value=3000, value=1, step=1)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Small Business", "Salaried"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=5, value=1, step=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=1, max_value=6, value=1, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=25, value=1, step=1)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", ["0", "1"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=0, max_value=5, value=3)
Designation = st.selectbox("Designation", ["Executive", "AVP", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=500000, value=30000, step=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchased" if prediction == 1 else "Package Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
