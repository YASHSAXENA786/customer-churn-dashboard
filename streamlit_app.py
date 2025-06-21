import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load model
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found. Please train and save the model as '{path}' before running the app.")
        st.stop()
    return joblib.load(path)

model = load_model(os.path.join(os.path.dirname(__file__), 'churn_model.pkl'))

# Feature list (excluding 'customerID' and 'Churn')
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Categorical options (from dataset)
categorical_options = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

st.title('Telco Customer Churn Prediction')
tabs = st.tabs(["Home", "Predict", "Insights"])

with tabs[0]:
    st.markdown("""
    # What is Customer Churn?
    **Customer churn** refers to when a customer stops using a company's service. In telecom, this means a subscriber cancels their phone, internet, or TV service.

    # Why Predicting Churn Matters
    - **High churn rates** can lead to significant revenue loss for telecom companies.
    - Acquiring new customers is much more expensive than retaining existing ones.
    - By predicting which customers are likely to churn, companies can take proactive steps (like special offers or improved service) to retain them.

    # How This Dashboard Helps
    - Uses **machine learning** to predict the likelihood of churn for individual customers based on their profile and usage.
    - Provides **visual insights** into the data and churn patterns.
    - Empowers decision-makers to design targeted retention strategies, reduce churn, and improve profitability.
    """)

with tabs[1]:
    st.header('Predict Churn')
    with st.form('churn_form'):
        user_input = {}
        for feat in features:
            if feat in categorical_options:
                user_input[feat] = st.selectbox(f"{feat}", categorical_options[feat])
            elif feat in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                user_input[feat] = st.number_input(f"{feat}", min_value=0.0, step=1.0)
        submitted = st.form_submit_button('Predict')
    if submitted:
        # Prepare input for model (Label Encoding as in training)
        input_df = pd.DataFrame([user_input])
        from sklearn.preprocessing import LabelEncoder
        for col in input_df.columns:
            if col in categorical_options:
                le = LabelEncoder()
                le.fit(categorical_options[col])
                input_df[col] = le.transform(input_df[col])
        proba = model.predict_proba(input_df)[0][1]
        percent = int(proba * 100)
        st.markdown(f"### Churn Probability: {percent}%")
        st.progress(percent)
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {'axis': {'range': [0, 100]}}))
        st.plotly_chart(fig)
        if percent > 60:
            st.warning('High risk of churn detected!')
            st.markdown('**Suggested Retention Offers:**')
            st.markdown('- Provide a personalized discount or loyalty reward\n- Offer a contract extension with added benefits\n- Assign a dedicated customer support representative\n- Survey for feedback and address specific pain points')

with tabs[2]:
    st.header('Churn Data Insights')
    data_path = os.path.join(os.path.dirname(__file__), 'telco_churn.csv')
    df = pd.read_csv(data_path)
    # Sidebar filters
    gender_filter = st.sidebar.multiselect('Filter by Gender', options=df['gender'].unique(), default=list(df['gender'].unique()))
    contract_filter = st.sidebar.multiselect('Filter by Contract Type', options=df['Contract'].unique(), default=list(df['Contract'].unique()))
    internet_filter = st.sidebar.multiselect('Filter by Internet Service', options=df['InternetService'].unique(), default=list(df['InternetService'].unique()))
    # Apply filters
    filtered_df = df[
        df['gender'].isin(gender_filter) &
        df['Contract'].isin(contract_filter) &
        df['InternetService'].isin(internet_filter)
    ]
    # Metrics summary
    total_customers = len(filtered_df)
    churn_rate = filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    avg_monthly_charges = filtered_df['MonthlyCharges'].mean()
    avg_tenure = filtered_df['tenure'].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Customers', f"{total_customers}")
    col2.metric('Churn Rate', f"{churn_rate:.2f}%")
    col3.metric('Avg. Monthly Charges', f"${avg_monthly_charges:.2f}")
    col4.metric('Avg. Tenure (months)', f"{avg_tenure:.1f}")
    # Import and call EDA function
    from churn_model import eda_plots_streamlit, top_churn_profiles
    eda_plots_streamlit(filtered_df, st)
    # Show top 5 likely to churn
    from churn_model import load_and_clean_data, top_churn_profiles, generate_csv_report, generate_pdf_report
    import joblib
    model = joblib.load('churn_model.pkl')
    top_churn_profiles(filtered_df, model, st, n=5)
    # Download report buttons
    st.markdown('---')
    st.markdown("""
    ### Business Insight
    The company is losing **26% of customers** due to short contracts and high bills. We recommend offering bundled packages to increase tenure and reduce churn. Focusing on customers with month-to-month contracts and higher monthly charges can have the greatest impact on retention.
    """)
    # Download report buttons
    # Add churn probability if not present
    if 'Churn_Probability' not in filtered_df.columns:
        from sklearn.preprocessing import LabelEncoder
        df_enc = filtered_df.copy()
        # Drop customerID if present
        if 'customerID' in df_enc.columns:
            df_enc = df_enc.drop('customerID', axis=1)
        categorical_cols = df_enc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Churn':
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        X = df_enc.drop('Churn', axis=1)
        filtered_df['Churn_Probability'] = model.predict_proba(X)[:,1]
    # CSV
    csv = generate_csv_report(filtered_df)
    st.download_button('Download CSV Report', data=csv, file_name='churn_report.csv', mime='text/csv')
    # PDF
    pdf = generate_pdf_report(filtered_df)
    st.download_button('Download PDF Report', data=pdf, file_name='churn_report.pdf', mime='application/pdf')
