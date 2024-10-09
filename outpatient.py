import streamlit as st

import joblib
import pandas as pd

# Load the trained model and label encoder
import joblib
from sklearn.preprocessing import LabelEncoder
df = pd.read_excel('/content/outpatienthr.xlsx')

# Assuming you have already encoded your labels as follows
label_encoder = LabelEncoder()

# Fit and transform diagnosis labels
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Save the label encoder to a file for later use in deployment
joblib.dump(label_encoder, 'label_encoder.pkl')

# Save the model
from sklearn.ensemble import RandomForestClassifier

# Label Encoding for categorical variables (like 'Gender')
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # Converts 'M' to 1 and 'F' to 0 (or vice versa)

# Assume 'features' are the columns used for prediction and 'target' is the label (e.g., diagnosis)
X = df[['age in years', 'sex', 'weight', 'height']]  # Features
y = df['diagnosis']  # Target

# Define the model
model = RandomForestClassifier()

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'diagnosis_model.pkl')

joblib.dump(model, 'diagnosis_model.pkl')

model = joblib.load('diagnosis_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create a title and description for the web app
st.title("Predict the Most Common Diagnosis")
st.write("This app predicts the diagnosis based on patient information such as age, sex, weight, and height.")

# Input fields for user data
age = st.number_input('Age in years', min_value=0, max_value=100, value=25)
sex = st.selectbox('Sex', ['Female', 'Male'])
weight = st.number_input('Weight (kg)', min_value=0.0, max_value=200.0, value=70.0)
height = st.number_input('Height (cm)', min_value=0.0, max_value=250.0, value=170.0)

# Convert the 'sex' input to numerical (Female -> 0, Male -> 1)
sex = 0 if sex == 'Female' else 1

# Button to trigger prediction
if st.button('Predict Diagnosis'):
    # Prepare input data for the model
    input_data = pd.DataFrame([[age, sex, weight, height]], columns=['age in years', 'sex', 'weight', 'height'])
    
    # Make a prediction using the model
    prediction = model.predict(input_data)
    
    # Decode the prediction to get the actual diagnosis label
    diagnosis = label_encoder.inverse_transform(prediction)
    
    # Display the prediction result
    st.success(f"The predicted diagnosis is: {diagnosis[0]}")

