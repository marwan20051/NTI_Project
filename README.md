# Credit Card Fraud Detection

An NTI machine-learning project that compares four classification models on a highly imbalanced real-world credit-card transaction dataset. The goal is to improve fraud detection through imbalance-aware training, hyperparameter tuning, and decision-threshold optimization.

## Project Status

**Phase:** Repository setup and experiment design

The team has selected the dataset, models, and evaluation strategy. Data exploration and model implementation are the next milestones.

## Problem Statement

Fraudulent transactions are rare, so a model can achieve very high accuracy while failing to identify fraud. This project focuses on detecting the minority fraud class and measuring the trade-off between missed fraud and false alarms.

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud/data) published by the Machine Learning Group at ULB.

- 284,807 transactions
- 492 fraudulent transactions
- Fraud rate of approximately 0.172%
- `V1` through `V28`: anonymized PCA-transformed numerical features
- `Time`: seconds elapsed since the first recorded transaction
- `Amount`: transaction value
- `Class`: target (`0` for legitimate and `1` for fraud)

The dataset is not stored in this repository. Download `creditcard.csv` from Kaggle and place it in the local `data/raw/` directory after the project structure is added.

## Models

The experiment will compare:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. CatBoost

Each model will have a reproducible baseline and an improved version. Improvements may include class weighting, training-only resampling, bounded hyperparameter tuning, and validation-only threshold optimization.

## Evaluation

The primary ranking metric is **area under the precision-recall curve (PR-AUC)**. Accuracy is not suitable as the main metric because of the extreme class imbalance.

The final comparison will also report:

- Precision, recall, and F1-score
- ROC-AUC
- Confusion matrices
- Training and inference time
- Estimated false-negative and false-positive cost

The test set will remain untouched until model settings and decision thresholds are fixed.

## Planned Workflow

1. Validate and explore the dataset.
2. Create reproducible train, validation, and test partitions.
3. Train baseline versions of all four models.
4. Apply imbalance-aware improvements and tune each model.
5. Select decision thresholds using validation data only.
6. Compare final results on the untouched test set.
7. Explain the winning model and analyze its errors.

## Reproducibility Rules

- Never commit the Kaggle dataset or generated model files.
- Fit preprocessing and resampling only on training data.
- Use fixed random seeds.
- Record package versions, parameters, split indices, and experiment results.
- Perform team work through branches and pull requests.

## Collaboration

Contributors should create a focused branch for each task, open a pull request, and request review before merging into `main`. Participation remains visible in the commit history and repository contributor insights.

## Roadmap

- [x] Select the project topic and dataset
- [x] Select the four candidate models
- [x] Define the evaluation strategy
- [ ] Add the Python project structure and dependencies
- [ ] Add data validation and exploratory analysis
- [ ] Train baseline models
- [ ] Improve and tune the models
- [ ] Produce the final comparison and report

## Scope and Limitations

The dataset covers two days of anonymized European card transactions from September 2013. Results therefore do not establish production readiness or generalization to current banking systems. The anonymized PCA features also limit business-level interpretation.
