import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_test_data():
    data = pd.read_csv("data/heart.csv")
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def get_all_runs():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("heart_disease_classification")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )
    return runs

def plot_model_comparison(metrics_df):
    # Plot metrics comparison for all models
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', width=0.8)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def evaluate_models():
    runs = get_all_runs()
    
    # Collect results from all models
    results = []
    for run in runs:
        model_name = run.data.tags.get('mlflow.runName', 'Unknown')
        metrics = {
            'Model': model_name,
            'Accuracy': run.data.metrics['accuracy'],
            'Precision': run.data.metrics['precision'],
            'Recall': run.data.metrics['recall'],
            'F1 Score': run.data.metrics['f1_score']
        }
        results.append(metrics)
    
    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results)
    metrics_df.set_index('Model', inplace=True)
    
    # Print comparison table
    print("\n=== Model Performance Comparison ===")
    print("\nMetrics Table:")
    print(metrics_df)
    
    # Plot comparison
    plot_model_comparison(metrics_df)
    
    # Find best model for each metric
    best_models = {
        'Accuracy': metrics_df['Accuracy'].idxmax(),
        'Precision': metrics_df['Precision'].idxmax(),
        'Recall': metrics_df['Recall'].idxmax(),
        'F1 Score': metrics_df['F1 Score'].idxmax()
    }
    
    print("\nBest Model for Each Metric:")
    for metric, model in best_models.items():
        print(f"{metric}: {model} ({metrics_df.loc[model, metric]:.4f})")

if __name__ == "__main__":
    try:
        evaluate_models()
    except Exception as e:
        print(f"Error: {str(e)}") 