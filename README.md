# Heart Disease Classification with MLflow

This project implements multiple machine learning models to predict heart disease using MLflow for experiment tracking and model management.

## Project Overview

The project uses the UCI Heart Disease dataset to train and compare different machine learning models for heart disease prediction. It includes:
- Multiple model implementations (Random Forest, SVM, KNN, etc.)
- MLflow tracking for experiment management
- Model performance comparison and visualization
- Automated model evaluation

## Dataset

The dataset is from the UCI Machine Learning Repository and includes various features related to heart disease diagnosis:
- 13 clinical features
- Binary classification target (presence/absence of heart disease)
- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## Models Implemented

1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Logistic Regression

## Results

### Model Performance Comparison

```
                     Accuracy  Precision    Recall  F1 Score
Model
Logistic Regression  0.866667   0.833333  0.833333  0.833333
KNN                  0.833333   0.818182  0.750000  0.782609
SVM                  0.883333   0.869565  0.833333  0.851064
Gradient Boosting    0.800000   0.730769  0.791667  0.760000
Random Forest        0.883333   0.840000  0.875000  0.857143
```

### Visualization
![Model Performance Comparison](model_comparison.png)

### Best Models by Metric
- **Accuracy**: SVM & Random Forest (0.8833)
- **Precision**: SVM (0.8696)
- **Recall**: Random Forest (0.8750)
- **F1 Score**: Random Forest (0.8571)

## Project Structure
```
project_root/
├── data/
│   └── heart.csv
├── src/
│   ├── train.py
│   └── evaluate.py
├── mlruns/
│   └── (MLflow tracking files)
├── README.md
└── results.md
```

## Setup and Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn mlflow seaborn matplotlib
```

## Usage

1. Train models:
```bash
python src/train.py
```

2. Evaluate models:
```bash
python src/evaluate.py
```

3. View MLflow UI:
```bash
mlflow ui
```

## Model Configurations

### Random Forest
- n_estimators: 100
- max_depth: 10
- random_state: 42

### Gradient Boosting
- n_estimators: 100
- max_depth: 5
- random_state: 42

### SVM
- kernel: rbf
- random_state: 42

### KNN
- n_neighbors: 5

### Logistic Regression
- default parameters
- random_state: 42

## Key Findings

1. **Top Performing Models**:
   - SVM and Random Forest tied for highest accuracy (88.33%)
   - SVM achieved best precision (86.96%)
   - Random Forest led in recall (87.50%) and F1 Score (85.71%)

2. **Model Characteristics**:
   - SVM: Excellent at minimizing false positives
   - Random Forest: Best at minimizing false negatives
   - Logistic Regression: Balanced performance across metrics
   - Gradient Boosting: Room for improvement with tuning
   - KNN: Simple but effective baseline model

## MLflow Integration

The project uses MLflow to:
- Track experiments and parameters
- Log performance metrics
- Store trained models
- Generate comparison visualizations
- Enable model versioning and reproduction

## Future Improvements

1. Model Enhancement:
   - Implement hyperparameter tuning
   - Add cross-validation
   - Explore ensemble methods

2. Feature Engineering:
   - Feature selection analysis
   - Feature importance ranking
   - Advanced preprocessing techniques

3. Evaluation:
   - Add stratified k-fold validation
   - Include ROC curves
   - Detailed confusion matrices
## Acknowledgments
- UCI Machine Learning Repository for the dataset
- MLflow team for the experiment tracking framework
