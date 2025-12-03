# Customer Churn Prediction with XGBoost and SHAP

This project uses a telecom customer dataset (`train.csv` and `test.csv`) to build a machine learning model that predicts whether a customer will churn (leave the service) or not.

The notebook covers the full workflow:
- Data loading and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature encoding  
- Training an XGBoost classifier  
- Evaluating model performance  
- Explaining the model using SHAP  
- Applying the final model on a test dataset  

---

## Project Structure


### 1. Setup and Imports

- Installs and imports all required libraries:
  - pandas, numpy  
  - seaborn, matplotlib  
  - scikit-learn (train_test_split, LabelEncoder, metrics, RandomizedSearchCV)  
  - xgboost  
  - shap  
- These libraries are used for:
  - Data manipulation  
  - Visualization  
  - Model building and tuning  
  - Model interpretation  

---

### 2. Loading the Training Data

- Reads the training dataset:

  ```python
  CSV_file = pd.read_csv('/content/train.csv')
