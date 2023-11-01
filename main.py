import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy import stats


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

    # Evaluate each algorithm using cross-validation
    CV_results = []
    for name, model in algorithms:
        scores = cross_val_score(model, train_features, train_target, cv=5, scoring="neg_mean_squared_error")
        CV_results.append((name, scores))
    print(f"cross validation results= {CV_results}")

    # Perform paired t-test to determine the best algorithm
    best_algs = ""
    best_p_value = 1.0

    for i in range(len(CV_results)):
        for j in range(i + 1, len(CV_results)):
            _, scores1 = CV_results[i]
            _, scores2 = CV_results[j]
            _, p_value = stats.ttest_rel(scores1, scores2)
            # print(stats.ttest_rel(scores1, scores2))
            if p_value < best_p_value:
                best_p_value = p_value
                best_algs = CV_results[i][0] + " vs. " + CV_results[j][0]

    # Comparing mean squared errors using paired t-test to find the best algorithm:
    print("The best algorithms are:", best_algs)

    # Train and evaluate the best algorithm on the test set
    best_model = None
    for name, model in algorithms:
        if name in best_algs:
            best_model = model
            break

    best_model.fit(train_features, train_target)
    mse = np.mean((best_model.predict(test_features) - test_target) ** 2)
    print(f"Mean Squared Error for the Best Algorithm on Test Data: {mse}")