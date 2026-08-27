import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aqueous Solubility Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.stMetric {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363d;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

FEATURES = [
    "MolLogP",
    "MolWt",
    "NumRotatableBonds",
    "AromaticProportion"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# ============================================================
# RDKit DESCRIPTOR CALCULATION
# ============================================================

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            "Invalid SMILES. Please enter a valid molecular structure."
        )

    mol_log_p = Descriptors.MolLogP(mol)

    mol_wt = Descriptors.MolWt(mol)

    rotatable_bonds = Lipinski.NumRotatableBonds(mol)

    heavy_atoms = mol.GetNumHeavyAtoms()

    if heavy_atoms > 0:

        aromatic_atoms = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetIsAromatic()
        )

        aromatic_proportion = aromatic_atoms / heavy_atoms

    else:

        aromatic_proportion = 0.0

    return {
        "MolLogP": mol_log_p,
        "MolWt": mol_wt,
        "NumRotatableBonds": rotatable_bonds,
        "AromaticProportion": aromatic_proportion
    }


# ============================================================
# SOLUBILITY CLASSIFICATION
# ============================================================

def classify_solubility(log_s):

    if log_s > -2:

        return "High Solubility"

    elif log_s > -4:

        return "Moderate Solubility"

    else:

        return "Poor Solubility / Insoluble"


# ============================================================
# RULE OF 5
# ============================================================

def calculate_rule_of_5(mol):

    molecular_weight = Descriptors.MolWt(mol)
    log_p = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = 0

    if molecular_weight > 500:
        violations += 1

    if log_p > 5:
        violations += 1

    if hbd > 5:
        violations += 1

    if hba > 10:
        violations += 1

    return {
        "Molecular Weight": molecular_weight,
        "LogP": log_p,
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
        "Violations": violations
    }


# ============================================================
# HEADER
# ============================================================

st.title("🧬 Aqueous Solubility Prediction Platform")

st.markdown(
    """
    **Machine-learning powered prediction of aqueous solubility (logS)**
    using molecular descriptors derived from chemical structures.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Control Panel")

mode = st.sidebar.radio(
    "Operating Mode",
    [
        "🔬 Single Molecule",
        "📂 Batch Screening",
        "📊 Model Insights",
        "ℹ️ About"
    ]
)


# ============================================================
# SINGLE MOLECULE
# ============================================================

if mode == "🔬 Single Molecule":

    st.header("🔬 Single Molecule Prediction")

    st.write(
        "Enter a molecule using its SMILES representation. "
        "RDKit will automatically calculate the descriptors used by the model."
    )

    examples = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ethanol": "CCO",
        "Benzene": "C1=CC=CC=C1"
    }

    selected_example = st.selectbox(
        "Load Example Molecule",
        ["Custom"] + list(examples.keys())
    )

    if selected_example != "Custom":

        default_smiles = examples[selected_example]

    else:

        default_smiles = ""

    smiles = st.text_input(
        "SMILES",
        value=default_smiles,
        placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O"
    )

    predict_button = st.button(
        "🚀 Predict Solubility",
        use_container_width=True
    )

    if predict_button:

        if not smiles.strip():

            st.warning("Please enter a SMILES string.")

        else:

            try:

                # --------------------------------------------
                # Create molecule
                # --------------------------------------------

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:

                    st.error("❌ Invalid SMILES.")
                    st.stop()

                # --------------------------------------------
                # Calculate descriptors
                # --------------------------------------------

                descriptors = calculate_descriptors(smiles)

                input_df = pd.DataFrame(
                    [descriptors],
                    columns=FEATURES
                )

                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = model.predict(input_df)

                log_s = float(prediction[0])

                classification = classify_solubility(log_s)

                # --------------------------------------------
                # Results
                # --------------------------------------------

                st.success("✅ Prediction completed successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Predicted logS",
                    f"{log_s:.3f}"
                )

                col2.metric(
                    "Solubility Class",
                    classification
                )

                col3.metric(
                    "Molecular Weight",
                    f"{descriptors['MolWt']:.2f} g/mol"
                )

                # --------------------------------------------
                # Molecular Structure
                # --------------------------------------------

                st.subheader("🧪 Molecular Structure")

                try:

                    from rdkit.Chem import Draw

                    image = Draw.MolToImage(
                        mol,
                        size=(450, 300)
                    )

                    st.image(
                        image,
                        caption="2D Molecular Structure"
                    )

                except Exception:

                    st.info(
                        "Molecular structure visualization unavailable."
                    )

                # --------------------------------------------
                # Descriptor Table
                # --------------------------------------------

                st.subheader("📐 Molecular Descriptors")

                descriptor_display = pd.DataFrame({
                    "Descriptor": FEATURES,
                    "Value": [
                        descriptors["MolLogP"],
                        descriptors["MolWt"],
                        descriptors["NumRotatableBonds"],
                        descriptors["AromaticProportion"]
                    ]
                })

                st.dataframe(
                    descriptor_display,
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------------
                # Rule of 5
                # --------------------------------------------

                st.subheader("💊 Lipinski Rule of 5")

                rule5 = calculate_rule_of_5(mol)

                r1, r2, r3, r4, r5 = st.columns(5)

                r1.metric(
                    "MW",
                    f"{rule5['Molecular Weight']:.1f}"
                )

                r2.metric(
                    "LogP",
                    f"{rule5['LogP']:.2f}"
                )

                r3.metric(
                    "HBD",
                    rule5["H-Bond Donors"]
                )

                r4.metric(
                    "HBA",
                    rule5["H-Bond Acceptors"]
                )

                r5.metric(
                    "Violations",
                    rule5["Violations"]
                )

                if rule5["Violations"] == 0:

                    st.success(
                        "✅ No Lipinski Rule-of-5 violations detected."
                    )

                else:

                    st.warning(
                        f"⚠️ {rule5['Violations']} Rule-of-5 violation(s) detected."
                    )

                # --------------------------------------------
                # Descriptor Visualization
                # --------------------------------------------

                st.subheader("📊 Molecular Profile")

                chart_data = pd.DataFrame({
                    "Descriptor": [
                        "MolLogP",
                        "MolWt",
                        "Rotatable Bonds",
                        "Aromatic Proportion"
                    ],

                    "Normalized Value": [
                        descriptors["MolLogP"] / 10,
                        descriptors["MolWt"] / 500,
                        descriptors["NumRotatableBonds"] / 20,
                        descriptors["AromaticProportion"]
                    ]
                })

                fig = px.bar(
                    chart_data,
                    x="Descriptor",
                    y="Normalized Value",
                    title="Normalized Molecular Descriptor Profile",
                    range_y=[0, 1]
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # --------------------------------------------
                # Interpretation
                # --------------------------------------------

                st.subheader("🧠 Prediction Interpretation")

                st.write(
                    f"""
                    The model predicts a **logS of {log_s:.3f}** for this molecule.

                    Based on the configured classification thresholds,
                    this corresponds to **{classification}**.

                    The prediction is generated from the four molecular
                    descriptors used during model training:
                    MolLogP, molecular weight, rotatable bonds, and
                    aromatic proportion.
                    """
                )

            except Exception as e:

                st.error(
                    f"❌ Prediction failed: {str(e)}"
                )


# ============================================================
# BATCH SCREENING
# ============================================================

elif mode == "📂 Batch Screening":

    st.header("📂 High-Throughput Batch Screening")

    st.write(
        """
        Upload a CSV containing either:

        **Option 1 — descriptors**

        `MolLogP, MolWt, NumRotatableBonds, AromaticProportion`

        **Option 2 — SMILES**

        `SMILES`
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("📄 Input Dataset")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            # ==================================================
            # SMILES MODE
            # ==================================================

            if "SMILES" in df.columns:

                st.info(
                    "SMILES column detected. Calculating molecular descriptors..."
                )

                descriptor_rows = []
                valid_rows = []

                for index, smiles in enumerate(df["SMILES"]):

                    try:

                        descriptor = calculate_descriptors(
                            str(smiles)
                        )

                        descriptor_rows.append(descriptor)
                        valid_rows.append(True)

                    except Exception:

                        descriptor_rows.append({
                            feature: np.nan
                            for feature in FEATURES
                        })

                        valid_rows.append(False)

                descriptor_df = pd.DataFrame(
                    descriptor_rows
                )

                result_df = pd.concat(
                    [
                        df.reset_index(drop=True),
                        descriptor_df
                    ],
                    axis=1
                )

                result_df["Valid_SMILES"] = valid_rows

                valid_mask = result_df["Valid_SMILES"]

                if valid_mask.any():

                    predictions = model.predict(
                        result_df.loc[
                            valid_mask,
                            FEATURES
                        ]
                    )

                    result_df.loc[
                        valid_mask,
                        "Predicted_logS"
                    ] = predictions

                    result_df.loc[
                        valid_mask,
                        "Solubility_Class"
                    ] = result_df.loc[
                        valid_mask,
                        "Predicted_logS"
                    ].apply(
                        classify_solubility
                    )

                st.success(
                    f"Processed {valid_mask.sum()} / {len(result_df)} molecules."
                )

            # ==================================================
            # DESCRIPTOR MODE
            # ==================================================

            elif all(
                feature in df.columns
                for feature in FEATURES
            ):

                result_df = df.copy()

                predictions = model.predict(
                    result_df[FEATURES]
                )

                result_df["Predicted_logS"] = predictions

                result_df["Solubility_Class"] = (
                    result_df["Predicted_logS"]
                    .apply(classify_solubility)
                )

                st.success(
                    "Batch prediction completed successfully!"
                )

            else:

                st.error(
                    "CSV must contain either a `SMILES` column "
                    "or all four required descriptor columns."
                )

                st.stop()

            # ==================================================
            # RESULTS
            # ==================================================

            st.subheader("📊 Prediction Results")

            st.dataframe(
                result_df,
                use_container_width=True
            )

            # ==================================================
            # DOWNLOAD
            # ==================================================

            csv_output = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_output,
                file_name="solubility_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"❌ Unable to process CSV: {str(e)}"
            )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif mode == "📊 Model Insights":

    st.header("📊 Model Architecture & Insights")

    st.subheader("🤖 Model")

    st.write(
        f"""
        **Algorithm:** Linear Regression

        **Number of features:** {getattr(model, "n_features_in_", "Unknown")}

        **Training feature names:** {getattr(
            model,
            "feature_names_in_",
            FEATURES
        )}
        """
    )

    # ========================================================
    # MODEL COEFFICIENTS
    # ========================================================

    if hasattr(model, "coef_"):

        st.subheader("📈 Model Coefficients")

        coefficients = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": model.coef_
        })

        st.dataframe(
            coefficients,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            coefficients,
            x="Feature",
            y="Coefficient",
            title="Linear Regression Feature Coefficients"
        )

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # INTERCEPT
    # ========================================================

    if hasattr(model, "intercept_"):

        st.metric(
            "Model Intercept",
            f"{float(model.intercept_):.4f}"
        )

    # ========================================================
    # FORMULA
    # ========================================================

    if hasattr(model, "coef_"):

        st.subheader("🧮 Model Equation")

        equation = (
            f"logS = {float(model.intercept_):.4f}"
        )

        for feature, coefficient in zip(
            FEATURES,
            model.coef_
        ):

            equation += (
                f" {'+' if coefficient >= 0 else '-'} "
                f"{abs(coefficient):.4f} × {feature}"
            )

        st.code(equation)


# ============================================================
# ABOUT
# ============================================================

else:

    st.header("ℹ️ About This Project")

    st.markdown(
        """
        ### 🧬 Aqueous Solubility Predictor

        This application predicts aqueous solubility expressed as
        **logS** using a machine-learning regression model.

        ### Machine Learning

        - Linear Regression
        - Delaney Aqueous Solubility dataset
        - Molecular descriptors as model features

        ### Molecular Features

        - MolLogP
        - Molecular Weight
        - Number of Rotatable Bonds
        - Aromatic Proportion

        ### Technology Stack

        - Python
        - Scikit-learn
        - Streamlit
        - RDKit
        - Pandas
        - NumPy
        - Plotly

        ### Important

        Predictions are model estimates and should not be treated
        as experimental measurements.
        """
    )

    st.info(
        "💡 The model should be evaluated on an appropriate held-out "
        "test set before being used for scientific decision-making."
    )