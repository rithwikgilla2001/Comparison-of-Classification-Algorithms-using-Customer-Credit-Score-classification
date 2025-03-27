import streamlit as st
import numpy as np
import pickle
import xgboost

# Load the trained model
with open("credit_score_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the exact features used in training
feature_names = ['Month', 'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
                 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                 'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
                 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                 'Payment_of_Min_Amount', 'Total_EMI_per_month',
                 'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance']

# Define categorical feature options
occupation_options = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 
                      'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']

credit_mix_options = ['Good', 'Standard', 'Bad', 'Non Standard']
payment_min_options = ['Yes', 'No']
payment_behavior_options = [
    'High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
    'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments',
    'High_spent_Medium_value_payments', 'Other_payments',
    'High_spent_Large_value_payments'
]
months = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

loan_types = ['Not Specified', 'Credit-Builder Loan', 'Personal Loan', 'Debt Consolidation Loan',
              'Student Loan', 'Payday Loan', 'Mortgage Loan', 'Auto Loan', 'Home Equity Loan']




# Streamlit UI
st.title("📊 Credit Score Classification App")
st.write("Enter the required details to predict your credit score category.")

# Create input fields for all features
user_inputs = {}

# Dropdown for Month
user_inputs['Month'] = st.selectbox("Select Month", months)

# Numeric Inputs
numeric_fields = [
    ('Age', 18, 100, 1), ('Annual_Income', 0.0, None, 500.0),
    ('Monthly_Inhand_Salary', 0.0, None, 100.0), ('Num_Bank_Accounts', 0, None, 1),
    ('Num_Credit_Card', 0, None, 1), ('Interest_Rate', 0.0, None, 0.1),
    ('Num_of_Loan', 0, None, 1), ('Delay_from_due_date', 0, None, 1),
    ('Num_of_Delayed_Payment', 0, None, 1), ('Changed_Credit_Limit', -100.0, None, 1.0),
    ('Num_Credit_Inquiries', 0, None, 1), ('Outstanding_Debt', 0.0, None, 100.0),
    ('Credit_Utilization_Ratio', 0.0, None, 0.1), ('Credit_History_Age', 0, None, 1),
    ('Total_EMI_per_month', 0.0, None, 50.0), ('Amount_invested_monthly', 0.0, None, 50.0),
    ('Monthly_Balance', -10000.0, None, 50.0)
]

for field, min_val, max_val, step in numeric_fields:
    user_inputs[field] = st.number_input(field, min_value=min_val, step=step)

# Categorical Inputs
user_inputs['Occupation'] = st.selectbox("Occupation", occupation_options)
user_inputs['Type_of_Loan'] = st.selectbox("Type_of_Loan", loan_types)
# user_inputs['Type_of_Loan'] = st.text_input("Type of Loan (comma-separated if multiple)", "")
user_inputs['Credit_Mix'] = st.selectbox("Credit Mix", credit_mix_options)
user_inputs['Payment_of_Min_Amount'] = st.selectbox("Payment of Minimum Amount", payment_min_options)
user_inputs['Payment_Behaviour'] = st.selectbox("Payment Behaviour", payment_behavior_options)

# Prediction Button
if st.button("Predict Credit Score"):
    try:
        # Encoding categorical values
        encoding_map = {
            'Occupation': {
                'Accountant': 0.876631, 'Architect': 0.908851, 'Developer': 0.893750, 'Doctor': 0.902558,
                'Engineer': 0.879079, 'Entrepreneur': 0.872443, 'Journalist': 0.910343, 'Lawyer': 0.900507,
                'Manager': 0.896455, 'Mechanic': 0.861718, 'Media_Manager': 0.922859, 'Musician': 0.906486,
                'Scientist': 0.872331, 'Teacher': 0.879646, 'Writer': 0.839626, '_______': 0.600000
            },
            'Credit_Mix': {'Good': 3, 'Non Standard': 1, 'Standard': 2, 'Bad': 0},
            'Payment_of_Min_Amount': {'Yes': 1, 'No': 0},
            'Type_of_Loan': {loan_type: i+1 for i, loan_type in enumerate(loan_types)},  # Map loan types to numerical values
            'Payment_Behaviour': {
                'High_spent_Large_value_payments': 0.888378, 'High_spent_Medium_value_payments': 0.888197,
                'High_spent_Small_value_payments': 0.888434, 'Low_spent_Large_value_payments': 0.888285,
                'Low_spent_Medium_value_payments': 0.888462, 'Low_spent_Small_value_payments': 0.888241,
                'Other_payments': 0.888117
            },
            'Month': {m: i+1 for i, m in enumerate(months)}
        }

        # Convert inputs
        processed_inputs = []
        for feature in feature_names:
            if feature in encoding_map:  # Categorical features
                processed_inputs.append(encoding_map[feature].get(user_inputs[feature], 0))
            elif feature == "Type_of_Loan":  # Handle loan types (adjust with the new loan types)
                loan_type = user_inputs[feature]
                processed_inputs.append(encoding_map['Type_of_Loan'].get(loan_type, 0))  # Ensure 'Other' is default
            else:  # Numeric features
                processed_inputs.append(user_inputs[feature])

        # Convert to NumPy array
        input_data = np.array(processed_inputs).reshape(1, -1)

        # Define mapping for predictions
        credit_score_mapping = {0: "Poor", 1: "Standard", 2: "Good"}
        # Debugging: Print shape
        # st.write(f"Input Shape: {input_data.shape}")  # Should be (1, 23)

        # Make prediction
        prediction = model.predict(input_data)
        # Convert numerical output to category
        predicted_category = credit_score_mapping.get(prediction[0], "Unknown")

        # Display result
        st.success(f"Predicted Credit Score Category: {predicted_category}")

    except Exception as e:
        st.error(f"Error: {e}")
