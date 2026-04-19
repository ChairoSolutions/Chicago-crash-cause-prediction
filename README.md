# Chicago-crash-cause-prediction
Predicting primary contributory causes of traffic crashes using machine learning.
## REPRODUCIBILITY
This is a guide on how to reproduce the analysis and model result found in this repository. Please follow the steps below
1. Clone the repository 
First clone this project to your local machine using gitbash
git clone https://github.com/ChairoSolutions/Chicago-crash-cause-prediction/tree/master
2. Setup an environment
The project requires python. Use a virtual environment to avoid library errors
The requirements.txt contains the specific versions of our libraries
pip install -r requirements.txt
3. Libraries 
The following are python libraries that are required for this analysis
- Loading and exploring the datasets - Pandas, numpy and scipy
- Machine learning and evaluation - scikitlearn 
- Interpretability - Shap
- Visualization - matpotlib and seaborn
- Deployment - Joblib
4. Data
Our data is sourced from the Chicago Data Portal. You should download the three csv datasets and place them in the data folder. The link attached has Traffic crashes, Vehicles and people
https://drive.google.com/drive/folders/1L_vsEDUWbOdqdFWgnFicctt0_qf8XXB9?usp=drive_link
5. Target Variable 
We used PRIM_CONTRIBUTORY_CAUSE 
 


## PROJECT FLOW
1. Business Understanding - Problem statement, goal, objectives, identifying our stakeholders
2. Data understanding - to gain insights in the data we are using
3. Data Processing - Selecting a target variable
4. Feature engineering - Selecting the features that strongl affect our target variable, ignoring features that have high cardinality and contribute to overfitting of our modes 
5. Pipeline - train test split, cleaning the datasets, normalizing our numerical features and encoding the categorical features
6. Modeling- Building different models 
7. Evaluation- Building a reusable code for evaluation and using different metrics to identify which model performs better on both training and testing 
8. Deployment - Deploting our model for our stake holders

# BUSINESS UNDERSTANDING
Chicago has been experiencing an uplift in traffic accidents. The city aims to eliminate fatalities but the data collected is very biased due to multiple post crash occurences. The project aims to develop a machine learning model that predicts the primary contributory cause of traffic crashes in Chicago, enabling stakeholders to identify high-risk conditions and design targeted interventions to improve road safety.


## Objectives
- Build a classification model to predict crash causes
- Identify key factors contributing to traffic accidents
- Provide insights that can help reduce accidents

##  Stakeholders
- Chicago City Planners → prioritize infrastructure improvements (e.g., intersections, lighting)
- Traffic Safety Authorities → design targeted safety campaigns and enforcement strategies
- Policy Makers → implement data-driven regulations to reduce high-risk crash scenarios

## Analytical Approach
This problem is framed as a **multi-class classification task**, where the target variable is:

PRIM_CONTRIBUTORY_CAUSE

## Success Criteria
The success of the model will be evaluated using:
- Macro F1-score (to handle class imbalance)
- Confusion matrix (to understand misclassifications)
- Interpretability (feature importance analysis)
- Baseline comparison (e.g., majority class or simple model)

## Key Challenges
- Imbalanced target classes
- High number of categorical variables
- Data spread across multiple tables (crashes, vehicles, people)
- Risk of data leakage: Some variables may contain information that is only known after the crash occurs (e.g., injury severity, damage estimates). These must be carefully excluded to avoid data leakage and ensure the model reflects real-world prediction scenarios.