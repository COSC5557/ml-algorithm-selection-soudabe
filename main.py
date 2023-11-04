import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def split_data(df, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    return train_test_split(df, test_size=test_size, random_state=random_state)


def separate_features_target(df, target_column):
    # Separate features and target variable
    features = df.drop([target_column], axis=1)
    target = df[target_column].copy()
    return features, target


if __name__ == "__main__":
    # Load the dataset
    data_frame = pd.read_csv("winequality-white.csv", sep=";")

    # Split the dataset into training and testing sets
    train_set, test_set = split_data(data_frame)

    # Separate features and target variable in the training and testing sets
    train_features, train_target = separate_features_target(train_set, target_column="quality")
    test_features, test_target = separate_features_target(test_set, target_column="quality")

    # Define a list of algorithms with default hyperparameters
    algorithms = [
        ("Linear Regression", LinearRegression()),
        ("Logistic Regression", LogisticRegression(max_iter=10000)),
        ("Support Vector Machine", SVR()),
        ("Random Forest", RandomForestRegressor(random_state=42)),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42))
    ]

    # Evaluate and train each algorithm
    results = []
    for name, model in algorithms:
        # Perform cross-validation
        scores = -cross_val_score(model, train_features, train_target, scoring="neg_mean_squared_error", cv=5)
        rmse_cv = np.sqrt(scores.mean())

        # Train the model on the full training set
        model.fit(train_features, train_target)

        # Predict on the test data
        y_pred = model.predict(test_features)
        rmse_test = np.sqrt(mean_squared_error(test_target, y_pred))

        # Return evaluation results
        results.append({
            "model_name": name,
            "rmse_cv": rmse_cv,
            "rmse_test": rmse_test
        })

    # Display evaluation results
    for result in results:
        print("Algorithm:", result["model_name"])
        print("RMSE for 5-fold CV:", result["rmse_cv"])
        print("RMSE for validation:", result["rmse_test"])
        print()

    # Create a bar plot to visualize the RMSE results
    model_names = [result["model_name"] for result in results]
    rmse_cv_scores = [result["rmse_cv"] for result in results]
    rmse_test_scores = [result["rmse_test"] for result in results]

    bar_width = 0.35
    index = np.arange(len(model_names))

    plt.figure(figsize=(12, 6))
    bar1 = plt.bar(index, rmse_cv_scores, bar_width, label='RMSE (5-fold CV)')
    bar2 = plt.bar(index + bar_width, rmse_test_scores, bar_width, label='RMSE (validation)')

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('RMSE Comparison of Different Models')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
