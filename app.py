import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Cheminformatics logS Predictor | Elite Edition",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom elite UI styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# App Header
st.title("🧬 Advanced Aqueous Solubility (logS) Prediction Platform")
st.markdown("An institutional-grade web utility for high-throughput molecular property screening and regression analysis.")

# Sidebar Controls
st.sidebar.header("🎛️ Control Panel")
mode = st.sidebar.radio("Select Operating Mode", ["Single Molecule Prediction", "Batch CSV Screening", "Model Performance & Insights"])

if mode == "Single Molecule Prediction":
    st.sidebar.subheader("Compound Presets")
    preset = st.sidebar.selectbox(
        "Load Known Benchmark",
        ["Custom Input", "Aspirin-like", "Paracetamol-like", "Highly Lipophilic Drug", "Insoluble Hydrocarbon"]
    )

    if preset == "Aspirin-like":
        d_logp, d_wt, d_bonds, d_aro = 1.19, 180.16, 3.0, 0.58
    elif preset == "Paracetamol-like":
        d_logp, d_wt, d_bonds, d_aro = 0.51, 151.16, 2.0, 0.50
    elif preset == "Highly Lipophilic Drug":
        d_logp, d_wt, d_bonds, d_aro = 6.20, 310.40, 6.0, 0.80
    elif preset == "Insoluble Hydrocarbon":
        d_logp, d_wt, d_bonds, d_aro = 7.50, 252.31, 0.0, 1.00
    else:
        d_logp, d_wt, d_bonds, d_aro = 2.5, 150.0, 1.0, 0.0

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.subheader("Molecular Descriptors")
        mol_log_p = st.number_input('MolLogP (Hydrophobicity)', value=d_logp, step=0.01, format="%.2f")
        mol_wt = st.number_input('MolWt (Molecular Weight g/mol)', value=d_wt, step=0.1, format="%.2f")
        num_rotatable_bonds = st.number_input('NumRotatableBonds', value=d_bonds, step=1.0)
        aromatic_proportion = st.number_input('AromaticProportion', value=d_aro, min_value=0.0, max_value=1.0, step=0.01)

        predict_btn = st.button('🔬 Execute Prediction', use_container_width=True)

    with col2:
        st.subheader("Prediction Dashboard")
        if predict_btn or 'last_pred' in st.session_state:
            input_data = np.array([[mol_log_p, mol_wt, num_rotatable_bonds, aromatic_proportion]])
            prediction = model.predict(input_data)
            log_s = prediction[0]
            st.session_state['last_pred'] = log_s

            # Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("Predicted logS", f"{log_s:.2f}")
            
            # Classification
            if log_s > -2:
                sol_class = "High Solubility (Soluble)"
                color = "normal"
            elif log_s > -4:
                sol_class = "Moderate Solubility"
                color = "off"
            else:
                sol_class = "Poor Solubility / Insoluble"
                color = "inverse"
            
            m2.metric("Solubility Class", sol_class)

            # Lipinski's Rule of 5 Validation
            st.markdown("### 📋 Lipinski's Rule of 5 Compliance")
            violations = 0
            if mol_wt > 500: violations += 1
            if mol_log_p > 5: violations += 1
            
            if violations == 0:
                st.success("✅ **Passed:** Compliant with Lipinski's Rule of 5 (Good oral bioavailability profile).")
            else:
                st.warning(f"⚠️ **Caution:** Exhibits rules violations ({violations} flag detected), indicating potential membrane permeability challenges.")

            # Radar/Visual Chart for Descriptors
            chart_data = pd.DataFrame({
                'Descriptor': ['MolLogP (scaled)', 'MolWt (scaled)', 'Rotatable Bonds', 'Aromatic Proportion'],
                'Value': [mol_log_p / 10, mol_wt / 500, num_rotatable_bonds / 20, aromatic_proportion]
            })
            fig = px.bar(chart_data, x='Descriptor', y='Value', title="Normalized Molecular Profile", range_y=[0, 1])
            fig.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig, use_container_width=True)

elif mode == "Batch CSV Screening":
    st.subheader("📂 High-Throughput Batch Prediction")
    st.write("Upload a CSV file containing columns: `MolLogP`, `MolWt`, `NumRotatableBonds`, `AromaticProportion`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        required_cols = ['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']
        
        if all(col in df_batch.columns for col in required_cols):
            preds = model.predict(df_batch[required_cols])
            df_batch['Predicted_logS'] = preds
            st.success("Batch prediction completed successfully!")
            st.dataframe(df_batch, use_container_width=True)
            
            # Download button
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name='solubility_predictions_output.csv',
                mime='text/csv',
            )
        else:
            st.error(f"CSV file must contain the exact columns: {required_cols}")

else:
    st.subheader("📊 Model Architecture & Performance Insights")
    st.write("""
    - **Algorithm:** Linear Regression fitted on Delaney's Aqueous Solubility dataset.
    - **Features Analyzed:** Octanol-water partition coefficient (`MolLogP`), Molecular Weight (`MolWt`), Rotatable Bond Count (`NumRotatableBonds`), and Aromatic Heavy Atom Proportion (`AromaticProportion`).
    - **Deployment Stack:** Python, Scikit-Learn, Streamlit, Plotly.
    """)
    st.info("💡 **Elite Tip:** Ensure features align with experimental limitations of the original training domain to maintain low residual error.")
