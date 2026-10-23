import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection System 🚀")

def user_input_features():
    st.subheader("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input('Transaction Amount', min_value=0.0, step=0.01)
        city_pop = st.number_input('City Population', min_value=0)
        trans_month = st.selectbox('Transaction Month', list(range(1, 13)))
        gender_option = st.selectbox('Gender', ('Male', 'Female'))
        gender = 0 if gender_option == 'Male' else 1
        age = st.number_input('Age', min_value=0, max_value=100)

    with col2:
        trans_year = st.number_input('Transaction Year', min_value=1950, max_value=2100)
        lat = st.number_input('Latitude', step=0.0001)
        long = st.number_input('Longitude', step=0.0001)
        category_options = [
            'SELECT',
            'Food and Dining', 'Gas and Transport', 'Grocery (Net)',
            'Grocery (POS)', 'Health and Fitness', 'Home',
            'Kids and Pets', 'Miscellaneous (Net)', 'Miscellaneous (POS)',
            'Personal Care', 'Shopping (Net)', 'Shopping (POS)',
            'Travel'
        ]
        selected_category = st.selectbox('Transaction Category', category_options, index=0)

    return {
        'amount': amount,
        'gender': gender,
        'lat': lat,
        'long': long,
        'city_pop': city_pop,
        'age': age,
        'trans_month': trans_month,
        'trans_year': trans_year,
        'selected_category': selected_category,
        'category_options': category_options
    }

user_inputs = user_input_features()

st.markdown("---")

if st.button('Predict 🚀'):

    selected_category = user_inputs['selected_category']
    if selected_category == 'SELECT':
        selected_category = 'Miscellaneous (POS)'

    data = {cat: 0 for cat in user_inputs['category_options'] if cat != 'SELECT'}
    data[selected_category] = 1

    data.update({
        'amount': user_inputs['amount'],
        'gender': user_inputs['gender'],
        'lat': user_inputs['lat'],
        'long': user_inputs['long'],
        'city_pop': user_inputs['city_pop'],
        'age': user_inputs['age'],
        'trans_month': user_inputs['trans_month'],
        'trans_year': user_inputs['trans_year'],
    })

    input_df = pd.DataFrame([data])

    # Add missing columns (if any)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Transaction is FRAUDULENT!")
    
    else:
        st.success(f"✅ Transaction is NOT FRAUDULENT!")
# Show sample data to help users understand the input format
st.markdown("### 🧾 Sample Data for Reference")
sample_data = pd.DataFrame([
    [4.97, 'Female', 36.0788, -81.1781, 3495, 30, 1, 2019,'Miscellaneous (Net)'],
    [107.23, 'Female', 48.8878, -118.2105, 149, 40, 1, 2019,'Grocery (POS)'],
    [220.11, 'Male', 42.1808, -112.2620, 4154, 56, 1, 2019,'Miscellaneous (POS)'],
    [45.00, 'Male', 46.2306, -112.1138, 1939, 51, 1, 2019,'Gas and Transport'],
    [41.96, 'Male', 38.4207, -79.4629, 99, 32, 1, 2020,'Miscellaneous (POS)'],
    [281.06, 'Male', 35.9946, -81.7266, 885, 30, 1, 2019,'Grocery (POS)'],
    [337.05, 'Female', 40.3359, -79.6607, 1472, 77, 1, 2019,'Grocery (POS)'],
    [276.31, 'Male', 29.4400, -98.4590, 1595797, 58, 1, 2019,'Grocery (POS)'],
    [942.62, 'Male', 35.9946, -81.7266, 885, 30, 1, 2019,'Shopping (Net)'],
    [275.73, 'Male', 29.4400, -98.4590, 1595797, 58, 1, 2019,'Grocery (POS)']
], columns=[
    'amount', 'gender', 'lat', 'long', 'city_pop', 'age', 'trans_month', 'trans_year','category'
])

st.dataframe(sample_data)