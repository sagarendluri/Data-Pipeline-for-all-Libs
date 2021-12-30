from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def lasso_model(X_train, y_train, X_test, y_test):
    lasso = Lasso(alpha=1.012)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    train_accuracy = lasso.score(X_train, y_train)
    test_accuracy = lasso.score(X_test, y_test)
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    print(f"Mean Squared Error: {mse}")
    print(f"Model Coefficients: {lasso.coef_}\n")


def ridge_model(X_train, y_train, X_test, y_test):
    ridge = Ridge(alpha=0.02, normalize=True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    train_accuracy = ridge.score(X_train, y_train)
    test_accuracy = ridge.score(X_test, y_test)
    print("train_accuracy,test_accuracy", train_accuracy, test_accuracy)
    print(f"Mean Squared Error: {mse}")
    print(f"Model Coefficients: {ridge.coef_}\n")
