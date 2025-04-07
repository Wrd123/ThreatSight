# ThreatSite: Cybersecurity Threat Detection Dashboard

## Quick Start Guide

Welcome to ThreatSite! Follow these simple steps to execute and use the application effectively.

### Step 1: Download Data Set
1. Visit (https://www.kaggle.com/datasets/cybersecurity-attacks)
2. Click the Download button at the top right corner
3. Scroll down to "Download dataset as zip" and click it (you may need to log in or create an account)
4. After downloading, unzip the file and ensure the `cybersecurity_attacks.csv` file is accessible in your desired folder

### Step 2: Run Application
1. Go to the ThreatSite Dashboard (https://threatsight.streamlit.app)
2. Once the webpage loads, click the "Browse Files" button on the right side of the screen
3. Locate and upload the previously downloaded `cybersecurity_attacks.csv` file

### Step 3: Using the Dashboard
After successfully uploading the file, the interactive dashboard will activate.

#### Exploration Features:
- Adjust the Anomaly Score Threshold using the provided slider
- Explore data insights using the interactive plots:
  - Heat-maps for anomaly score distributions by Attack Type
  - Time-based Anomaly Detection for both Day of the Week & Hourly Distribution
  - Scatter plots, Density Heat-maps and Violin Plots for relationships like Packet Length vs. Anomaly Scores
  - Drill down into specific high-anomaly events for detailed analysis

#### Model Interaction and Feedback Dashboard:
- Hyper-parameter tuning allowing adjustment of the RandomForest ensemble
- Feature selection allowing the use of all features or only a specific targeted set

## Local Development
If you're looking to run or modify the application locally:
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
4. Follow the same steps as above to upload and analyze data

## About the Data
The Kaggle Cyber Security Attacks dataset contains synthesized cybersecurity event data including:
- Attack signatures and types
- Anomaly scores
- Packet information
- Severity levels
- Temporal data

## Features
- Data Ingestion and Processing: Clean and transform raw security data
- Interactive Visualizations: Dynamic, adjustable visualizations for pattern recognition
- Model Interaction: Train and evaluate machine learning models with customizable parameters
- Drill-Down Analysis: Detailed examination of individual security events