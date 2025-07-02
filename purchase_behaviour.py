

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("🧮 Applied Probability and Statistics in E-commerce")

# Upload dataset
uploaded_file = st.file_uploader("Upload Online Retail CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')

    # Data cleaning
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df.dropna(subset=['CustomerID'], inplace=True)

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm.replace([np.inf, -np.inf], np.nan, inplace=True)
    rfm.dropna(inplace=True)

    st.subheader("📊 Descriptive Statistics of RFM Features")
    st.write(rfm.describe())

    st.subheader("📈 RFM Feature Distributions")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(rfm['Recency'], bins=30, ax=ax[0]); ax[0].set_title("Recency")
    sns.histplot(rfm['Frequency'], bins=30, ax=ax[1]); ax[1].set_title("Frequency")
    sns.histplot(rfm['Monetary'], bins=30, ax=ax[2]); ax[2].set_title("Monetary")
    st.pyplot(fig)

    st.subheader("📐 Probability Distribution Fitting (Monetary)")
    mu, std = stats.norm.fit(rfm['Monetary'])
    st.write(f"Fitted Normal Distribution: mean = {mu:.2f}, std = {std:.2f}")

    plt.figure(figsize=(6, 4))
    sns.histplot(rfm['Monetary'], bins=50, kde=True)
    plt.title("Distribution of Monetary Value")
    st.pyplot()

    st.subheader("🔬 Hypothesis Testing")
    high_freq = rfm[rfm['Frequency'] > rfm['Frequency'].median()]['Monetary']
    low_freq = rfm[rfm['Frequency'] <= rfm['Frequency'].median()]['Monetary']
    t_stat, p_val = stats.ttest_ind(high_freq, low_freq, equal_var=False)
    st.write(f"T-test Statistic: {t_stat:.2f}")
    st.write(f"P-value: {p_val:.4f}")
    if p_val < 0.05:
        st.success("✅ High-frequency customers spend significantly more.")
    else:
        st.warning("❌ No significant difference in spending.")

    st.subheader("📏 Confidence Interval (95%)")
    ci_low, ci_high = stats.norm.interval(
        0.95,
        loc=rfm['Monetary'].mean(),
        scale=rfm['Monetary'].std() / np.sqrt(len(rfm))
    )
    st.write(f"95% Confidence Interval for Mean Monetary Value: ({ci_low:.2f}, {ci_high:.2f})")

    st.subheader("📉 Regression: Predict Monetary from Recency & Frequency")
    X = rfm[['Recency', 'Frequency']]
    y = rfm['Monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    st.write("Regression Coefficients:")
    st.json(dict(zip(['Recency', 'Frequency'], reg.coef_)))
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

    st.subheader("🔮 Predict Future Purchase Value")
    rec = st.number_input("Enter Recency (days):", min_value=1, value=10)
    freq = st.number_input("Enter Frequency (purchases):", min_value=1, value=5)
    sample = pd.DataFrame({'Recency': [rec], 'Frequency': [freq]})
    prediction = reg.predict(sample)[0]
    st.success(f"Predicted Monetary Value: ₹{prediction:.2f}")
