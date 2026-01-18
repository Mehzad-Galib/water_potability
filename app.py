import gradio as gr
import joblib
import numpy as np
import pandas as pd

# 1. Load the trained model pipeline
# Ensure 'water_model.pkl' is in the same directory
model = joblib.load('water_model.pkl')

# 2. Define the prediction function
def predict_potability(ph, hardness, solids, chloramines, sulfate, 
                        conductivity, organic_carbon, trihalomethanes, turbidity):
    
    # Calculate the engineered feature (Chemical_Balance) used during training
    # We add a tiny epsilon (1e-5) to avoid division by zero
    chemical_balance = chloramines / (sulfate + 1e-5)
    
    # Create a DataFrame with the exact column names used during training
    input_df = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity, chemical_balance
    ]], columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Chemical_Balance'
    ])
    
    # Get prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    if prediction == 1:
        return f"✅ Potable (Safe for Drinking)\nConfidence: {probability[1]:.2%}"
    else:
        return f"❌ Not Potable (Unsafe)\nConfidence: {probability[0]:.2%}"

# 3. Create the Gradio UI
interface = gr.Interface(
    fn=predict_potability,
    inputs=[
        gr.Slider(0, 14, value=7, label="pH Level"),
        gr.Number(value=200, label="Hardness (mg/L)"),
        gr.Number(value=20000, label="Solids (Total Dissolved Solids - ppm)"),
        gr.Number(value=7, label="Chloramines (ppm)"),
        gr.Number(value=333, label="Sulfate (mg/L)"),
        gr.Number(value=426, label="Conductivity (μS/cm)"),
        gr.Number(value=14, label="Organic Carbon (ppm)"),
        gr.Number(value=66, label="Trihalomethanes (μg/L)"),
        gr.Number(value=4, label="Turbidity (NTU)")
    ],
    outputs=gr.Textbox(label="Result"),
    title="🚰 Water Potability Prediction AI",
    description="Enter the water quality parameters below to predict if the water is safe for human consumption based on the Random Forest model.",
    theme="soft"
)

# 4. Launch the app
if __name__ == "__main__":
    interface.launch()