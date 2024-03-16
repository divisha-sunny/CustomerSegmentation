from sklearn import preprocessing
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'Models/final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Data/Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Credit Card Customer Segmentation")

with st.form("my_form"):
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff_Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Freqency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency,
             cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments,
             prc_full_payment, tenure]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust = loaded_model.predict(data)[0]
    st.write('Data Belongs to Cluster', clust)

    cluster_df1 = df[df['Cluster'] == clust]

    if not cluster_df1.empty:
        # 1) Average Balance in Text
        avg_balance = cluster_df1['BALANCE'].mean()
        st.subheader("Average Balance")
        st.markdown(f"<h3 style='font-size: 24px; color: skyblue;'>{avg_balance:.2f}</h3>", unsafe_allow_html=True)

        # 6) Sum of PURCHASE Transactions made
        purchases_sum = cluster_df1['PURCHASES_TRX'].sum()
        st.subheader("Sum of Purchase Transactions")
        st.markdown(f"<h3 style='font-size: 24px; color: skyblue;'>{purchases_sum}</h3>", unsafe_allow_html=True)

        # 7) Average amounts of payments done and Average(MINIMUM_PAYMENTS) done by user in text
        avg_payments = cluster_df1['PAYMENTS'].mean()
        avg_min_payments = cluster_df1['MINIMUM_PAYMENTS'].mean()
        st.subheader("Average Amount of Payments")
        st.markdown(f"<h3 style='font-size: 24px; color: skyblue;'>{avg_payments:.2f}</h3>", unsafe_allow_html=True)
        st.subheader("Average Minimum Payments")
        st.markdown(f"<h3 style='font-size: 24px; color: skyblue;'>{avg_min_payments:.2f}</h3>", unsafe_allow_html=True)



        # 2) Bar Graph of 'BALANCE_FREQUENCY' columns with 4 bins between 0 to 1.
        st.subheader("Distribution of the frequency of the Balance")
        plt.figure(figsize=(10, 6))
        bins = pd.cut(cluster_df1['BALANCE_FREQUENCY'], bins=4)
        bin_counts = bins.value_counts().sort_index()
        bin_labels = [f"{bin.left:.2f}-{bin.right:.2f}" for bin in bin_counts.index]
        plt.bar(bin_labels, bin_counts, color='skyblue')  # Set the color to midnight blue
        plt.xlabel('Balance Frequency')
        plt.ylabel('Count')
        st.pyplot()


        # 3) Histogram of PURCHASES
        st.subheader("Distribution of the Purchases")
        plt.figure(figsize=(10, 6))
        sns.histplot(cluster_df1, x='PURCHASES', kde=True, color='skyblue')
        plt.xlabel('Purchases')
        plt.ylabel('Count')
        st.pyplot()

        # 4) Bar graph with PURCHASES_FREQUENCY frequency with 4 bins between 0 and 1
        st.subheader("Distribution of the frequency of the Purchases")
        plt.figure(figsize=(10, 6))
        bins = pd.cut(cluster_df1['PURCHASES_FREQUENCY'], bins=4)
        bin_counts = bins.value_counts().sort_index()
        bin_labels = [f"{bin.left:.2f}-{bin.right:.2f}" for bin in bin_counts.index]
        plt.bar(bin_labels, bin_counts, color='skyblue')
        plt.xlabel('Purchase Frequency Bins')
        plt.ylabel('Count')
        st.pyplot()

        # 8) Bar graph of count of tenure
        st.subheader("Count of Tenure")
        plt.figure(figsize=(10, 6))
        tenure_count = cluster_df1['TENURE'].value_counts().sort_index()
        ax = tenure_count.plot(kind='bar', color='skyblue')
        ax.bar_label(ax.containers[0])
        plt.xlabel('Tenure')
        plt.ylabel('Count')
        st.pyplot()

    else:
        st.write("No data available for the selected cluster.")
