import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Change directory to where the data files are located
os.chdir('C:/Users/gchar/Major_Project')

# Load training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Map target variable
train.Loan_Status = train.Loan_Status.map({'Y': 1, 'N': 0})

# Check for missing values in training data
print(train.isnull().sum())

# Combine training and testing data for preprocessing
Loan_status = train.Loan_Status
train.drop('Loan_Status', axis=1, inplace=True)
Loan_ID = test.Loan_ID
data = pd.concat([train, test], ignore_index=True)

# Define a function to preprocess the data
def preprocess_data(data):
    data.Gender = data.Gender.map({'Male': 1, 'Female': 0})
    data.Married = data.Married.map({'Yes': 1, 'No': 0})
    data.Dependents = data.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
    data.Education = data.Education.map({'Graduate': 1, 'Not Graduate': 0})
    data.Self_Employed = data.Self_Employed.map({'Yes': 1, 'No': 0})
    data.Property_Area = data.Property_Area.map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    
    # Fill missing values
    data.Credit_History.fillna(data.Credit_History.mode()[0], inplace=True)
    data.Married.fillna(data.Married.mode()[0], inplace=True)
    data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
    data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.median(), inplace=True)
    data.Gender.fillna(data.Gender.mode()[0], inplace=True)
    data.Dependents.fillna(data.Dependents.mode()[0], inplace=True)
    data.Self_Employed.fillna(data.Self_Employed.mode()[0], inplace=True)
    
    return data

# Preprocess the data
data = preprocess_data(data)

# Drop Loan_ID column
data.drop('Loan_ID', axis=1, inplace=True)

# Split data into features and target variable
train_X = data.iloc[:614, :]
train_y = Loan_status

# Scale features
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_X_scaled, train_y, test_size=0.2, random_state=0)

# Define models and hyperparameters for tuning
models = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {'C': [0.1, 1, 10]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    },
    "SVC": {
        "model": SVC(),
        "params": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7]}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}
    },
    "Linear Discriminant Analysis": {
        "model": LinearDiscriminantAnalysis(),
        "params": {}
    }
}

# Train and evaluate models with hyperparameter tuning
results = []
names = []

for name, config in models.items():
    model = config['model']
    params = config['params']
    
    if params:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)
    
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result = cross_val_score(best_model, train_X_scaled, train_y, cv=kfold, scoring='accuracy')
    results.append(cv_result)
    names.append(name)
    print(f"{name}: {cv_result.mean():.6f}")

# Plot model performance
plt.figure(figsize=(12, 6))
plt.boxplot(results, labels=names)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

# Train the best model and save it
best_model_name = "Logistic Regression"  # Replace with the model you find best
best_model = models[best_model_name]['model'].fit(X_train, y_train)

with open(f'{best_model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Load the saved model
with open(f'{best_model_name.lower().replace(" ", "_")}_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Preprocess the test data
test_data = preprocess_data(test.drop('Loan_ID', axis=1))

# Scale test data
test_data_scaled = scaler.transform(test_data)

# Ensure the test data has the same columns in the same order as the training data
test_data_scaled = test_data_scaled[:, :train_X_scaled.shape[1]]

# Make predictions
predictions = loaded_model.predict(test_data_scaled)
print(predictions)
