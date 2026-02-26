"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np
import sklearn 

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """
    diabeties = load_diabetes()

    X = diabeties.data
    Y = diabeties.target

    X_train, x_test,  Y_train , y_test = train_test_split(
        X,Y,
        test_size = 0.2,
        random_state = 42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(x_test)


    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error( Y_train, y_train_pred)
    test_mse = mean_squared_error(y_test,y_test_pred)
    
    train_r2 = r2_score(Y_train,y_train_pred)
    test_r2 = r2_score(y_test,y_test_pred)
    
    top3 = np.argsort(np.abs(model.coef_))[::-1][:3].tolist()
    
    return train_mse, test_mse, train_r2, test_r2, top3





# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """

    
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


    pipeline = Pipeline([
        ('scaler', StandardScaler()),   # Step 1: Scale features
        ('model', LinearRegression())   # Step 2: Train model
    ])


    cv_scores = cross_val_score(
        pipeline,       # model pipeline
        X_train,        # training data only
        y_train,
        cv=5,           # 5-fold
        scoring='r2'    # evaluate using R²
    )


    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)


    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """
    # STEP 1: Load dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # STEP 3: Standardize features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train Logistic Regression
    model = LogisticRegression(max_iter=5000)

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # STEP 5: Compute metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    print("Confusion Matrix:")
    print(cm)

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1, 10, 100]
    results = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        results[C] = (train_acc, test_acc)

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=1, max_iter=5000))
    ])

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring='accuracy'
    )

    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    return mean_accuracy, std_accuracy
