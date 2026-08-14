import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Aqueous Solubility (logS) Predictor")
st.write("Enter the molecular descriptors below to predict solubility:")

# Input fields for your 4 features
mol_log_p = st.number_input('MolLogP', value=2.5)
mol_wt = st.number_input('MolWt', value=150.0)
num_rotatable_bonds = st.number_input('NumRotatableBonds', value=1.0)
aromatic_proportion = st.number_input('AromaticProportion', value=0.0)

# Prediction button
if st.button('Predict Solubility'):
    input_data = np.array([[mol_log_p, mol_wt, num_rotatable_bonds, aromatic_proportion]])
    prediction = model.predict(input_data)
    st.success(f"Predicted logS Solubility: {prediction[0]:.2f}")
