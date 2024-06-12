import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Function to extract stock data
def extract_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="max")
        stock_data.to_csv(f"{symbol}_stock_data.csv")
        st.success(f"Stock data saved to {symbol}_stock_data.csv")
    except Exception as e:
        st.error(f"Error extracting data for symbol {symbol}: {e}")

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_valid, Y_train, Y_valid):
    models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
    results = []

    for model in models:
        model.fit(X_train, Y_train)
        train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])
        valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:,1])
        results.append({'Model': type(model).__name__, 'Training Accuracy': train_auc, 'Validation Accuracy': valid_auc})

    return results

# Main function
def main():
    st.set_page_config(
        page_title="Stock Analysis Web App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )
    
    # Customized CSS styles
    st.markdown(
        """
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-image: url('https://classplusapp.com/growth/wp-content/uploads/2022/06/5-Easy-Steps-To-Create-Own-Stock-Market-Website-768x513.jpg');
                color: #333; /* Font color */
            }
            .navbar {
                background-color: #163548;
                padding: 10px;
                border-radius: 0;
                color: white;
                text-align: center;
                animation: fadeIn 1s;
            }
            .navbar h1 {
                color: white;
                font-family: 'Arial Black', sans-serif;
            }
            .navbar p {
                color: white;
                font-style: italic;
            }
            .navbar a {
                color: white;
                text-decoration: none;
                margin-right: 20px;
                padding: 8px 15px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .navbar a:hover {
                background-color: #14516e;
            }
            .home-content {
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 10px;
                animation: fadeInUp 1s;
            }
            .footer {
                background-color: rgba(22, 53, 72, 0.8);
                color: white;
                text-align: center;
                padding: 20px;
                border-radius: 0 0 10px 10px;
                animation: fadeIn 1s;
            }
            .footer a {
                color: white;
                text-decoration: none;
                margin: 0 10px;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            .progress-bar-container {
                width: 100%;
                margin: 20px auto;
                animation: fadeIn 1s;
            }
            .progress-bar {
                width: 100%;
                background-color: #e9ecef;
                border-radius: 5px;
                overflow: hidden;
            }
            .progress-bar-fill {
                height: 20px;
                background-color: #007bff;
                animation: fillAnimation 4s ease-in-out forwards;
            }
            @keyframes fillAnimation {
                from {
                    width: 0%;
                }
                to {
                    width: 100%;
                }
            }
            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header
    st.markdown(
        """
        <div class="navbar">
            <h1>Stock Analysis Web App</h1>
            <p>Explore stock data and visualize trends</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data", "Plots", "Model Training"])
    
    if page == "Home":
        st.markdown(
            """
            <div class="home-content">
                <h2>Welcome to Stock Analysis Web App</h2>
                <p>This web app allows you to explore stock data and visualize trends.</p>
                <button onclick="document.getElementById('user_input').focus();" style="background-color:#163548;color:white;border:none;border-radius:5px;padding:10px 20px;margin-top:20px;cursor:pointer;">Get Started</button>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif page == "Data":
        # User input for stock symbol
        user_main_symbol = st.sidebar.text_input("Enter a stock symbol:", key="user_input")
        
        if user_main_symbol:
            st.write(f"User input: {user_main_symbol}")
            extract_stock_data(user_main_symbol)
            
            # Read the CSV file
            df = pd.read_csv(f"{user_main_symbol}_stock_data.csv")
            
            # Convert 'Date' column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Display basic info and head of the dataframe
            st.write(df.info())
            st.dataframe(df.head())
    elif page == "Plots":
        # User input for stock symbol
        user_main_symbol = st.sidebar.text_input("Enter a stock symbol:", key="user_input")
        
        if user_main_symbol:
            st.write(f"User input: {user_main_symbol}")
            extract_stock_data(user_main_symbol)
            
            # Read the CSV file
            df = pd.read_csv(f"{user_main_symbol}_stock_data.csv")
            
            # Convert 'Date' column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Display a line plot of Close price with Date
            st.subheader("Close Price")
            plt.figure(figsize=(15, 5))
            plt.plot(df.index, df['Close'])
            plt.title('Close price', fontsize=15)
            plt.xlabel('Date')
            plt.ylabel('Price in dollars')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            st.pyplot(plt)
            
            # Display distribution plots
            st.subheader("Distribution Plots")
            plt.figure(figsize=(20, 10))
            for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                plt.subplot(2, 3, i + 1)
                sb.distplot(df[col])
            st.pyplot(plt)
            
            # Display box plots
            st.subheader("Box Plots")
            plt.figure(figsize=(20, 10))
            for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                plt.subplot(2, 3, i + 1)
                sb.boxplot(df[col])
            st.pyplot(plt)
    elif page == "Model Training":
        user_main_symbol = st.sidebar.text_input("Enter a stock symbol:", key="user_input")
        st.subheader("Model Training")
        st.write("Training models...")
        df = pd.read_csv(f"{user_main_symbol}_stock_data.csv")
            # Convert 'Date' column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
            # Assuming you have the features and target variables
        features = df[['open-close', 'low-high', 'is_quarter_end']]
        target = df['target']

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            features, target, test_size=0.1, random_state=2022)

        results = train_and_evaluate_models(X_train, X_valid, Y_train, Y_valid)

            # Display results
        for result in results:
            st.write(result)
    else:
        st.error("Please enter a stock symbol on the 'Data' page before training models.")

    # Progress bar
    with st.sidebar.expander("Progress"):
        st.write("Data processing...")
        st.write('<div class="progress-bar-container"><div class="progress-bar"><div class="progress-bar-fill"></div></div></div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>Explore more:</p>
            <a href="https://www.google.com/">Google</a>
            <a href="https://www.moneycontrol.com/stocksmarketsindia/">MoneyControl</a>
            <a href="http://www.nseindia.com/">NSE India</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()
