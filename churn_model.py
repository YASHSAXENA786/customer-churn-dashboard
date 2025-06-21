import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import io
import base64
from fpdf import FPDF

def load_and_clean_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Drop 'customerID' column
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert 'TotalCharges' to float, coerce errors to NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Handle missing values (fill with median for numeric, mode for categorical)
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Assume 'Churn' is the target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def eda_plots(csv_path):
    df = pd.read_csv(csv_path)

    # Class distribution of Churn
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn', data=df)
    plt.title('Class Distribution of Churn')
    plt.show()

    # Correlation heatmap (encode categorical for correlation)
    df_corr = df.copy()
    for col in df_corr.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Bar plot: Contract vs Churn
    if 'Contract' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='Contract', hue='Churn', data=df)
        plt.title('Contract Type vs Churn')
        plt.show()

    # Bar plot: tenure vs Churn
    if 'tenure' in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
        plt.title('Tenure vs Churn')
        plt.show()

    # Bar plot: MonthlyCharges vs Churn
    if 'MonthlyCharges' in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30)
        plt.title('Monthly Charges vs Churn')
        plt.show()

def train_random_forest(csv_path, model_path='rf_model.joblib'):
    X, y = load_and_clean_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    joblib.dump(clf, model_path)
    return clf, report

def train_and_save_rf(csv_path, model_path='churn_model.pkl'):
    X, y = load_and_clean_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    joblib.dump(clf, model_path)
    return clf, acc, report

def eda_plots_streamlit(df, st):
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Class distribution of Churn
    st.subheader('Class Distribution of Churn')
    fig1 = px.histogram(df, x='Churn', color='Churn',
        color_discrete_sequence=px.colors.sequential.RdBu,
        title='Class Distribution of Churn',
        hover_data=df.columns)
    fig1.update_layout(bargap=0.2, xaxis_title='Churn', yaxis_title='Count')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    - Most customers do not churn, but a significant minority do.
    - Churned users are a key focus for retention strategies.
    - Understanding their characteristics helps reduce revenue loss.
    """)

    # Correlation heatmap (encode all non-numeric for correlation)
    st.subheader('Correlation Heatmap')
    df_corr = df.copy()
    for col in df_corr.columns:
        if not np.issubdtype(df_corr[col].dtype, np.number):
            df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
    corr = df_corr.corr()
    fig2 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        colorbar=dict(title='Correlation'),
        hoverongaps=False))
    fig2.update_layout(title='Correlation Heatmap', xaxis_title='', yaxis_title='', autosize=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    - Tenure and contract type are strongly correlated with churn.
    - Monthly charges and total charges also show relationships with churn.
    - Categorical features like contract and payment method are important predictors.
    """)

    # Bar plot: Contract vs Churn
    if 'Contract' in df.columns:
        st.subheader('Contract Type vs Churn')
        fig3 = px.histogram(df, x='Contract', color='Churn', barmode='group',
            color_discrete_sequence=px.colors.sequential.RdBu,
            title='Contract Type vs Churn',
            hover_data=df.columns)
        fig3.update_layout(xaxis_title='Contract', yaxis_title='Count')
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        - Most churned users are on month-to-month contracts.
        - Annual and two-year contracts have much lower churn rates.
        - Consider offering discounts or incentives for longer-term contracts.
        """)

    # Bar plot: Tenure vs Churn
    if 'tenure' in df.columns:
        st.subheader('Tenure vs Churn')
        fig4 = px.histogram(df, x='tenure', color='Churn', nbins=30, barmode='overlay',
            color_discrete_sequence=px.colors.sequential.RdBu,
            title='Tenure vs Churn',
            hover_data=df.columns)
        fig4.update_layout(xaxis_title='Tenure', yaxis_title='Count')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        - Customers with shorter tenure are more likely to churn.
        - Retention efforts should focus on new customers in their first year.
        - Loyalty programs can help increase tenure and reduce churn.
        """)

    # Bar plot: MonthlyCharges vs Churn
    if 'MonthlyCharges' in df.columns:
        st.subheader('Monthly Charges vs Churn')
        fig5 = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30, barmode='overlay',
            color_discrete_sequence=px.colors.sequential.RdBu,
            title='Monthly Charges vs Churn',
            hover_data=df.columns)
        fig5.update_layout(xaxis_title='Monthly Charges', yaxis_title='Count')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""
        - Higher monthly charges are associated with higher churn rates.
        - Customers with lower charges are less likely to churn.
        - Consider targeted offers for high-paying customers at risk of churning.
        """)

def top_churn_profiles(df, model, st, n=5):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    # Encode categorical features as in training
    df_enc = df.copy()
    categorical_cols = df_enc.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Churn':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    # Predict churn probabilities
    X = df_enc.drop('Churn', axis=1)
    churn_probs = model.predict_proba(X)[:,1]
    df['Churn_Probability'] = churn_probs
    # Get top N likely to churn
    top_churn = df.sort_values('Churn_Probability', ascending=False).head(n)
    # Select key features to display
    display_cols = ['Churn_Probability', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'Contract', 'MonthlyCharges', 'InternetService']
    display_cols = [col for col in display_cols if col in top_churn.columns]
    with st.expander(f"Top {n} Customer Profiles Most Likely to Churn", expanded=True):
        # Format probability as percent string for display
        top_churn_disp = top_churn[display_cols].copy()
        top_churn_disp['Churn_Probability'] = (top_churn_disp['Churn_Probability'] * 100).round(2).astype(str) + '%'
        st.dataframe(top_churn_disp)

def generate_csv_report(df, filename='churn_report.csv'):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

def generate_pdf_report(df, filename='churn_report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Churn Prediction Report', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    # Add summary stats
    total_customers = len(df)
    churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    avg_monthly_charges = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()
    pdf.cell(0, 10, f'Total Customers: {total_customers}', ln=True)
    pdf.cell(0, 10, f'Churn Rate: {churn_rate:.2f}%', ln=True)
    pdf.cell(0, 10, f'Avg. Monthly Charges: ${avg_monthly_charges:.2f}', ln=True)
    pdf.cell(0, 10, f'Avg. Tenure: {avg_tenure:.1f} months', ln=True)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 5 Customers Most Likely to Churn:', ln=True)
    pdf.set_font('Arial', '', 10)
    # Add top 5 table
    display_cols = ['Churn_Probability', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'Contract', 'MonthlyCharges', 'InternetService']
    display_cols = [col for col in display_cols if col in df.columns]
    top5 = df.sort_values('Churn_Probability', ascending=False).head(5)
    for idx, row in top5.iterrows():
        pdf.cell(0, 8, f"{row[display_cols].to_dict()}", ln=True)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output
