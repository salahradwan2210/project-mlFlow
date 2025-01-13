import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Load data directly from UCI ML Repository
heart_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load and save data locally
data = pd.read_csv(heart_data_url, names=column_names, na_values='?')

# Handle missing values
data = data.dropna()

# Convert target to binary values (0 or 1)
data['target'] = data['target'].map(lambda x: 1 if x > 0 else 0)

# Save data locally
data.to_csv("data/heart.csv", index=False)

# Set experiment name
mlflow.set_experiment("heart_disease_classification")

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of models to try
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Save model
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}") 