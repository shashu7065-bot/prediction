import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load trained model
# ===========================
model = joblib.load("logistic_model.pkl")

# ===========================
# App Title
# ===========================
st.title("⚙️ Vehicle Predictive Maintenance System")

st.write("""
This app predicts **Machine Failure** based on sensor readings 🚗🔧  
Provide the machine parameters in the sidebar and click **Predict**.
""")

# ===========================
# Sidebar Inputs
# ===========================
st.sidebar.header("🔢 Input Machine Parameters")

air_temp = st.sidebar.number_input("Air temperature [K]", 300, 400, 350)
proc_temp = st.sidebar.number_input("Process temperature [K]", 300, 400, 350)
rpm = st.sidebar.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
torque = st.sidebar.number_input("Torque [Nm]", 0, 100, 40)
tool_wear = st.sidebar.number_input("Tool wear [min]", 0, 300, 50)

# Failure modes (binary inputs)
twf = st.sidebar.selectbox("Tool wear failure (TWF)", [0, 1])
hdf = st.sidebar.selectbox("Heat dissipation failure (HDF)", [0, 1])
pwf = st.sidebar.selectbox("Power failure (PWF)", [0, 1])
osf = st.sidebar.selectbox("Overstrain failure (OSF)", [0, 1])
rnf = st.sidebar.selectbox("Random failure (RNF)", [0, 1])

# Machine type
type_choice = st.sidebar.selectbox("Machine Type", ["H", "L", "M"])

# One-hot encoding (baseline H → 0,0)
type_L, type_M = 0, 0
if type_choice == "L":
    type_L = 1
elif type_choice == "M":
    type_M = 1

# ===========================
# Create Input DataFrame
# ===========================
input_df = pd.DataFrame([[
    air_temp, proc_temp, rpm, torque, tool_wear,
    twf, hdf, pwf, osf, rnf,
    type_L, type_M
]], columns=[
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
    "TWF", "HDF", "PWF", "OSF", "RNF",
    "Type_L", "Type_M"
])

# ===========================
# Prediction
# ===========================
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Machine Failure Predicted! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ No Failure — Machine is Healthy. (Failure Probability: {probability:.2f})")

# ===========================
# Extra Analysis (Optional)
# ===========================
st.markdown("---")
st.header("📊 Dataset Insights & Analysis")

df = pd.read_csv("ai4i2020.csv")
if "UDI" in df.columns:
    df = df.drop(columns=["UDI", "Product ID"])

# Failure distribution
st.subheader("Failure Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Machine failure", data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_encoded.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)
