AI-Powered Business Transformation: Optimizing Bidding and Proposals
This project implements an AI-powered decision support system for optimizing the Go/No-Go decision-making process in bidding and proposals. 
It uses a variety of machine learning models to analyze project-related features and provide actionable insights that assist in the bidding decision.

**Project Overview**
The system is designed to assist the Bidding & Proposals department by analyzing critical project data and making Go/No-Go recommendations. It leverages historical data on past projects and employs several machine learning models, including:

**1. Random Forest Classifier,
2. Logistic Regression,
3. XGBoost(Extreme Gradient Boosting),
4. Support Vector Machine,
5. K-Nearest Neighbors (KNN),
6. Neural Networks,
7. Naive Bayes.**

These models are trained to provide a comprehensive view of Project Scope Complexity,	Technical Feasibility,	Financial Feasibilit,y	Contract Terms,
Relevant Experience,	Internal Resources Available,	Risk Level,	
Market Demand,	Strategic Alignment and Long Term Impact.
**Key Features**
**Data Preprocessing:** Prepares and cleans data for model training.
**Multi-Model Evaluation:** Compares multiple models to determine the best-performing ones.
**Feature Importance Analysis:** Identifies critical project features using feature importance and logistic regression coefficients.
**Decision Support:** Provides Go/No-Go predictions based on selected project characteristics.
**Visualization:** Includes feature importance graphs for model interpretability.
**Flask Web Application:** Allows users to input data for real-time predictions on Go/No-Go outcomes.
**Project Structure**
**Datasets:** Contains the training and selected feature datasets (Dataset_Go_NoGo_full_data.csv, Dataset_Go_NoGo_selected_features.csv).
**Model Files:** Pre-trained models and scalers in .pkl format (e.g., knn_go_no_go.pkl, rf_go_no_go.pkl, scaler_go_no_go.pkl).
Scripts:
**App_Code.py:** Main application code for data processing, model training, and predictions.
**Model_Training.py:** Script for training and evaluating machine learning models.
**Static & Templates:** Flask application files for UI.
**Visualizations:** Feature importance plots for Random Forest, XGBoost, and Logistic Regression coefficients.
**Getting Started**
**Prerequisites**
Python 3.8+
Required Python libraries: scikit-learn, pandas, numpy, flask, matplotlib, xgboost
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/Malik7007/AI-Powered-Business-Transformation-Optimizing-Bidding-and-Proposals.git
cd AI-Powered-Business-Transformation-Optimizing-Bidding-and-Proposals
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Start the Flask web application:
bash
Copy code
python App_Code.py
Access the web application at http://127.0.0.1:5000/.

**Usage**
**Input Data:** Use the web interface to input a project information by selection option.
**Run Prediction:** The system will analyze the input information and provide Go/No-Go recommendations for each project.
**View Results:** The results page displays predictions for each machine learning model used.
**Model Training**
For retraining models, you can use the Model_Training.py script. Ensure datasets are placed correctly, and the required libraries are installed.
