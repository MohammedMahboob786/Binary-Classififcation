 # Hotel Booking Prediction Models


This GitHub repository contains a Jupyter Notebook for hotel booking prediction using Decision Trees, Random Forests, Gradient Boosting and XGBoost. The notebook loads `hotel.csv` data, preprocesses it, builds bagging and boosting model and makes predictions. Here's a step-by-step explanation of the code:

---

## Table of Contents

1. [Importing Libraries](#1-importing-libraries)
2. [Data Preprocessing](#2-data-preprocessing)
   - [Handling Missing Values](#handling-missing-values)
   - [Dropping Duplicates and Irrelevant Columns](#dropping-duplicates-and-irrelevant-columns)
   - [Feature Engineering](#feature-engineering)
   - [Changing Data Types](#changing-data-types)
   - [One-Hot Encoding](#one-hot-encoding)
3. [Feature Selection](#3-feature-selection)
4. [Model Training (without Hyperparameter Tuning)](#4-model-training-without-hyperparameter-tuning)
5. [Model Testing (without Hyperparameter Tuning)](#5-model-testing-without-hyperparameter-tuning)
   
## Bagging Model:

6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Model Training with Hyperparameter Tuning](#7-model-training-with-hyperparameter-tuning)
8. [Model Testing with Hyperparameter Tuning](#8-model-testing-with-hyperparameter-tuning)
9. [ROC Curve and AUC](#9-roc-curve-and-auc)
10. [Threshold Analysis](#10-threshold-analysis)
11. [Test Results](#11-test-results)

## Boosting Model:

12. [ROC Curve and AUC](#12-roc-curve-and-auc)
13. [Threshold Analysis](#13-threshold-analysis)
14. [Test Results](#14-test-results)

---

### 1. Importing Libraries:
Starts by importing necessary Python libraries for data manipulation, machine learning, and visualization. Key libraries include pandas, numpy, scikit-learn, matplotlib and seaborn.


### 2. Data Preprocessing:
- **Handling Missing Values:** Addresses missing values in certain columns ('company', 'children', 'country', 'agent') by applying strategies like dropping columns or filling missing values with medians or modes.
- **Dropping Duplicates and Irrelevant Columns:** Duplicate rows are removed, and columns deemed irrelevant ('reservation_status_date', 'index', 'reservation_status') are dropped.
- **Feature Engineering:** New features ('kids', 'Full_stay', 'Total_members') are created based on existing columns.
- **Changing Data Types:** Modifies the data types of certain columns to ensure consistency and accuracy.
- **One-Hot Encoding:** Categorical variables are one-hot encoded, converting them into a format suitable for machine learning models.


### 3. Feature Selection:
- Uses Recursive Feature Elimination (RFE) with the model to select important features for the model.


### 4. Model Training (without Hyperparameter Tuning):
- Splits the dataset into training and testing sets.
- Trains the model on the training set.


### 5. Model Testing (without Hyperparameter Tuning):
- Evaluates the trained model on both the training and testing sets using metrics such as confusion matrix, accuracy, precision, recall, and F1-score.
- Plots feature importances.

---
## Bagging Model:
---

### 6. Hyperparameter Tuning:
Conducts hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters for the RandomForestClassifier and DecisionTreeClassifier.

### 7. Model Training with Hyperparameter Tuning:
- Retrains the RandomForestClassifier and DecisionTreeClassifier with the best hyperparameters obtained from GridSearchCV.

### 8. Model Testing with Hyperparameter Tuning:
Evaluates the tuned model on both the training and testing sets.

### 9. ROC Curve and AUC
Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) are plotted to assess the model's performance.

### 10. Threshold Analysis:

- Performs analysis on different probability thresholds to find the optimum threshold for prediction.
- Plots the accuracy, sensitivity, and specificity for different threshold values.

### 11. Test Results:
Generates final predictions using a chosen threshold and evaluates the model's performance on the test set. The results include metrics such as accuracy, recall, precision, false positive rate (FPR), and specificity.

---
## Boosting Model:
---

### 12. ROC Curve and AUC
Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) are plotted to assess the model's performance.

### 13. Threshold Analysis:
- Performs analysis on different probability thresholds to find the optimum threshold for prediction.
- Plots the accuracy, sensitivity, and specificity for different threshold values.

### 14. Test Results:
Generates final predictions using a chosen threshold and evaluates the model's performance on the test set. The results include metrics such as accuracy, recall, precision, false positive rate (FPR), and specificity.

---
