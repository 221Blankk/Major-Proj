import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import yfinance as yf

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Function to extract stock data and generate the 'Target' column
def extract_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="max")
        # Generate 'Target' column as soon as data is fetched
        stock_data['Target'] = (stock_data['Close'].diff() > 0).astype(int)
        stock_data.to_csv(f"{symbol}_stock_data.csv")
        st.success(f"Stock data saved to {symbol}_stock_data.csv")
        return stock_data
    except Exception as e:
        st.error(f"Error extracting data for symbol {symbol}: {e}")
        return None

# Function to calculate confusion matrix
def calculate_confusion_matrix(actual_values, predicted_values):
    return confusion_matrix(actual_values, predicted_values)

# Function to calculate correlation matrix and plot heatmap
def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix > 0.9, annot=True, cbar=False)
    plt.title('Highly Correlated Features')
    plt.xticks(rotation=45)
    st.pyplot()

# Function to preprocess data and train models
def preprocess_and_train(df):
    # Check for necessary columns including 'Target'
    if 'Target' not in df.columns:
        st.error("DataFrame is missing the 'Target' column.")
        return

    # Additional data checks and preprocessing
    if 'Open-Close' not in df.columns:
        df['Open-Close'] = df['Close'] - df['Open']
    if 'Low-High' not in df.columns:
        df['Low-High'] = df['High'] - df['Low']
    if 'Is_Quarter_End' not in df.columns:
        df['Is_Quarter_End'] = 0

    # Define features and target
    features = df[['Open-Close', 'Low-High', 'Is_Quarter_End']]
    target = df['Target']

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

    # Define models
    models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

    # Train models and print evaluation metrics
    for model in models:
        model.fit(X_train, Y_train)
        train_acc = model.score(X_train, Y_train)
        valid_acc = model.score(X_valid, Y_valid)
        st.write(f'Model: {model.__class__.__name__}')
        st.write('Training Accuracy:', train_acc)
        st.write('Validation Accuracy:', valid_acc)

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
    page = st.sidebar.radio("Go to", ["Home", "Data", "Plots"])
    
    if page == "Home":
        st.markdown(
            """
            <div class="home-content">
                <h2>Welcome to Stock Analysis Web App</h2>
                <p>This web app allows you to explore stock data and visualize trends.</p>
                <button onclick="document.getElementById('user_input').focus();" style="background-color:#163548;color:white;border:none;border-radius:
                border-radius:5px;padding:10px 20px;margin-top:20px;cursor:pointer;">Get Started</button>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif page == "Data":
        # User input for stock symbol
        user_main_symbol = st.sidebar.text_input("Enter a stock symbol:", key="user_input")
        
        if user_main_symbol:
            df = extract_stock_data(user_main_symbol)
            
            if df is not None and 'Target' in df.columns:
                st.write("Data loaded and 'Target' column created successfully.")
                st.write(df.info())
                st.dataframe(df.head())
            else:
                st.error("Failed to load data or 'Target' column not created.")
            
    elif page == "Plots":
        user_main_symbol = st.sidebar.text_input("Enter a stock symbol for plots:", key="plot_input")
        
        if user_main_symbol:
            df = extract_stock_data(user_main_symbol)
            
            if df is not None and 'Target' in df.columns:
                # Close Price Line Plot
                st.subheader("Close Price")
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(df.index, df['Close'])
                ax.set_title('Close Price Trend', fontsize=15)
                ax.set_xlabel('Date')
                ax.set_ylabel('Close Price')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                # Confusion Matrix Plot
                st.subheader("Confusion Matrix")
                actual_values = np.random.randint(0, 2, size=len(df))
                predicted_values = np.random.randint(0, 2, size=len(df))
                cm = calculate_confusion_matrix(actual_values, predicted_values)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                # Distribution Plots
                st.subheader("Distribution Plots")
                fig, axs = plt.subplots(2, 3, figsize=(20, 10))
                columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for i, col in enumerate(columns):
                    sns.histplot(df[col], kde=True, ax=axs[i//3, i%3])
                    axs[i//3, i%3].set_title(f'Distribution of {col}')
                plt.tight_layout()
                st.pyplot(fig)

                # Box Plots
                st.subheader("Box Plots")
                fig, axs = plt.subplots(2, 3, figsize=(20, 10))
                for i, col in enumerate(columns):
                    sns.boxplot(y=df[col], ax=axs[i//3, i%3])
                    axs[i//3, i%3].set_title(f'Box Plot of {col}')
                plt.tight_layout()
                st.pyplot(fig)

                # Correlation Heatmap
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df.corr()
                sns.heatmap(corr_matrix, annot=True, cbar=True, ax=ax)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)

                # Model Training
                st.subheader("Model Training")
                preprocess_and_train(df)  # Assumes preprocess_and_train prints results directly


    
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