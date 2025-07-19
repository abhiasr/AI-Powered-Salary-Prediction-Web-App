import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder # You need LabelEncoder in app.py too! 

# Load the trained model
# Your model was saved directly after fitting on X_train (which was already LabelEncoded and scaled implicitly by the pipeline for other models, but not explicitly for the best_model, which is GradientBoostingClassifier here).
# Looking at your notebook, the 'best_model' is saved *without* the StandardScaler pipeline, meaning it expects raw, transformed (LabelEncoded) data. 
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# --- IMPORTANT: Instantiate and fit LabelEncoders for app.py ---
# You need to re-create the LabelEncoders with the *same* categories as during training.
# The best way to do this is to save them during training, or define them with known categories.
# For simplicity and demonstration, I'm re-creating them with common categories.
# If your training data had fewer/different categories, you MUST adjust these lists.

# Create LabelEncoder instances for each categorical feature
# We'll map the Streamlit string inputs to numerical labels
workclass_encoder = LabelEncoder()
# Fit with all possible categories your model might have seen during training
# IMPORTANT: These lists MUST match the unique values of your original 'asr.csv'
# after replacements (like '?' to 'Others') and removals ('Without-pay', 'Never-worked').
# Based on your notebook:
# data.workclass.replace({'?':'Others'},inplace=True) 
# data=data[data['workclass']!='Without-pay'] 
# data=data[data['workclass']!='Never-worked'] 
workclass_encoder.fit(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Others']) # 'Others' came from '?' 

marital_status_encoder = LabelEncoder()
marital_status_encoder.fit(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

occupation_encoder = LabelEncoder()
# Based on your notebook's replacement of '?' to 'Others' for occupation: 
occupation_encoder.fit([
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces", "Others" # 'Others' came from '?' 
])

relationship_encoder = LabelEncoder()
relationship_encoder.fit(['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])

race_encoder = LabelEncoder()
race_encoder.fit(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

gender_encoder = LabelEncoder()
gender_encoder.fit(['Male', 'Female'])

native_country_encoder = LabelEncoder()
# This list needs to be comprehensive from your training data,
# including any '?' replacement if you did that for 'native-country' (not shown in notebook but possible).
# For now, I'll use a standard broad list; adjust to your exact training data's unique values.
native_country_encoder.fit([
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico",
    "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy",
    "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
    "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "France", "Greece",
    "Ecuador", "Ireland", "Hong", "Trinadad&Tobago", "Thailand", "Laos",
    "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Honduras", "Hungary", "Scotland",
    "Holand-Netherlands", "?" # Include '?' if you didn't replace it, or its replacement if you did.
])

# Sidebar inputs (these must match your training feature columns and their order)
st.sidebar.header("Input Employee Details")

# ✨ IMPORTANT: Match your original dataset columns exactly after preprocessing in the notebook.
# The `experience` column is NOT in your original `asr.csv` or `x` DataFrame, so remove it.
# The `education` column was dropped, so use `educational-num` instead.

age = st.sidebar.slider("Age", 17, 75, 30) # Range adjusted based on your outlier removal 
workclass_str = st.sidebar.selectbox("Workclass", workclass_encoder.classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 0, 1500000, 200000)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10) # Range adjusted based on your outlier removal 
marital_status_str = st.sidebar.selectbox("Marital Status", marital_status_encoder.classes_)
occupation_str = st.sidebar.selectbox("Job Role", occupation_encoder.classes_)
relationship_str = st.sidebar.selectbox("Relationship", relationship_encoder.classes_)
race_str = st.sidebar.selectbox("Race", race_encoder.classes_)
gender_str = st.sidebar.selectbox("Gender", gender_encoder.classes_)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 50000, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Max hours often higher than 80
native_country_str = st.sidebar.selectbox("Native Country", native_country_encoder.classes_)


# Apply Label Encoding to the user inputs
workclass_encoded = workclass_encoder.transform([workclass_str])[0]
marital_status_encoded = marital_status_encoder.transform([marital_status_str])[0]
occupation_encoded = occupation_encoder.transform([occupation_str])[0]
relationship_encoded = relationship_encoder.transform([relationship_str])[0]
race_encoded = race_encoder.transform([race_str])[0]
gender_encoded = gender_encoder.transform([gender_str])[0]
native_country_encoded = native_country_encoder.transform([native_country_str])[0]


# Build input DataFrame (⚠️ MUST match features and order from your training 'x' DataFrame)
# The order here must be exactly the order of columns in your 'x' DataFrame from the notebook: 
# 'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_encoded],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status_encoded],
    'occupation': [occupation_encoded],
    'relationship': [relationship_encoded],
    'race': [race_encoded],
    'gender': [gender_encoded],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country_encoded]
})


st.write("### 🔎 Input Data (Processed)")
# Display the numerical input_df that the model will see
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    try:
        # The model directly expects the numerical values now
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}") # Output is already '<=50K' or '>50K'
    except Exception as e:
        st.error(f"Error during prediction: {e}. Please ensure all input features match the model's training data (names, order, and expected numerical types after encoding).")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # --- IMPORTANT: Apply the same preprocessing to batch_data ---
    # Define the columns that need Label Encoding
    categorical_cols = [
        'workclass', 'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'native-country'
    ]

    # Create a copy to avoid SettingWithCopyWarning
    processed_batch_data = batch_data.copy()

    # Apply Label Encoding
    for col in categorical_cols:
        if col in processed_batch_data.columns:
            # Need to handle potential unknown categories in uploaded batch file
            # If an unknown category appears, transform will raise an error.
            # A more robust solution involves saving the fitted encoders and using .transform()
            # with handle_unknown='ignore' if available (not directly in LabelEncoder).
            # For LabelEncoder, you might need to try-except or map to a default if unknown.
            # For simplicity here, assuming valid categories in uploaded CSV.
            if col == 'workclass':
                processed_batch_data[col] = workclass_encoder.transform(processed_batch_data[col])
            elif col == 'marital-status':
                processed_batch_data[col] = marital_status_encoder.transform(processed_batch_data[col])
            elif col == 'occupation':
                processed_batch_data[col] = occupation_encoder.transform(processed_batch_data[col])
            elif col == 'relationship':
                processed_batch_data[col] = relationship_encoder.transform(processed_batch_data[col])
            elif col == 'race':
                processed_batch_data[col] = race_encoder.transform(processed_batch_data[col])
            elif col == 'gender':
                processed_batch_data[col] = gender_encoder.transform(processed_batch_data[col])
            elif col == 'native-country':
                processed_batch_data[col] = native_country_encoder.transform(processed_batch_data[col])
        else:
            st.warning(f"Column '{col}' not found in uploaded CSV for batch processing. This might cause prediction errors.")

    # Drop 'education' if it exists in the uploaded batch data, as it was dropped during training. 
    if 'education' in processed_batch_data.columns:
        processed_batch_data = processed_batch_data.drop(columns=['education'])

    # Ensure the uploaded CSV has ALL the expected columns in the correct order for prediction.
    # This list of columns MUST EXACTLY match the 'x' DataFrame from your training notebook. 
    expected_columns_order = [
        'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country'
    ]

    missing_cols = [col for col in expected_columns_order if col not in processed_batch_data.columns]
    if missing_cols:
        st.error(f"The processed batch CSV is missing the following required columns: {', '.join(missing_cols)}")
    else:
        # Reorder columns to match the model's expectation
        processed_batch_data_ordered = processed_batch_data[expected_columns_order]

        st.write("Processed batch data preview (reordered):", processed_batch_data_ordered.head())
        try:
            batch_preds = model.predict(processed_batch_data_ordered)
            batch_data['PredictedClass'] = batch_preds # Add to the original batch_data for user download
            st.write("✅ Predictions:")
            st.write(batch_data.head())
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Error during batch prediction: {e}. Please ensure the uploaded CSV file contains all the necessary features and that their values are compatible with the model's expected encodings/ranges.")
