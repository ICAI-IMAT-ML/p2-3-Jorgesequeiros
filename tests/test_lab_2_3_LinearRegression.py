import pytest
import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add it to sys.path
sys.path.insert(0, project_root)
from src.lab_2_3_LinearRegression import *


def test_fit_simple():
    model = LinearRegressor()
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    model.fit_simple(X, y)

    assert (
        pytest.approx(model.coefficients, 1e-6) == 2.0
    ), "Simple linear regression coefficient is incorrect."
    assert (
        pytest.approx(model.intercept, 1e-6) == 0.0
    ), "Simple linear regression intercept is incorrect."


def test_fit_multiple():
    model = LinearRegressor()
    np.random.seed(42)
    X = np.random.rand(100, 3)
    true_coefficients = np.array([2.5, -1.5, 3.0])
    true_intercept = 1.0

    # Generate y with some noise
    y = true_intercept + X.dot(true_coefficients) + np.random.normal(0, 0.1, 100)

    model.fit_multiple(X, y)

    # Check if coefficients are reasonably close to true values
    assert pytest.approx(model.intercept, rel=0.1) == true_intercept
    assert np.allclose(model.coefficients, true_coefficients, rtol=0.1)


def test_predict_simple():
    model = LinearRegressor()
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    model.fit_simple(X, y)
    predictions = model.predict(np.array([6, 7]))

    assert predictions.shape == (
        2,
    ), "Prediction shape for simple regression is incorrect."
    assert pytest.approx(predictions[0], 1e-6) == 12.0, "Prediction for 6 is incorrect."
    assert pytest.approx(predictions[1], 1e-6) == 14.0, "Prediction for 7 is incorrect."


def test_predict_multiple():
    model = LinearRegressor()
    np.random.seed(42)
    X = np.random.rand(100, 3)
    true_coefficients = np.array([2.5, -1.5, 3.0])
    true_intercept = 1.0

    # Generate y with some noise
    y = true_intercept + X.dot(true_coefficients) + np.random.normal(0, 0.1, 100)

    model.fit_multiple(X, y)

    X_test = np.random.rand(20, 3)
    y_expected = true_intercept + X_test.dot(true_coefficients)

    # Predictions should be close to expected values (within noise level)
    y_pred = model.predict(X_test)
    assert np.allclose(y_pred, y_expected, rtol=0.1)


def test_predict_unfitted():
    model = LinearRegressor()
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.predict(np.array([1, 2, 3]))


def test_evaluate_regression_perfect():
    # Perfect predictions: metrics should be R2=1, RMSE=0, MAE=0.
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["R2"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["MAE"] == pytest.approx(0.0)


def test_evaluate_regression_imperfect():
    # Imperfect predictions.
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    # Calculated manually:
    # ss_res = 1.5, ss_tot ~29.1875 --> R2 ~ 0.94857
    # RMSE = sqrt(1.5/4) ~ 0.61237, MAE = 0.5
    expected_R2 = 1 - (1.5 / 29.1875)
    expected_RMSE = np.sqrt(1.5 / 4)
    expected_MAE = 0.5

    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["R2"] == pytest.approx(expected_R2, rel=1e-5)
    assert metrics["RMSE"] == pytest.approx(expected_RMSE, rel=1e-5)
    assert metrics["MAE"] == pytest.approx(expected_MAE, rel=1e-5)

def test_anscombe_quartet():
    anscombe, datasets, models, result = anscombe_quartet()
    assert len(datasets)== 4, "Datasets should contain 4 elements."
    assert all(
        map(lambda r2: pytest.approx(r2, 1e-2) == 0.666, result["R2"])
    ), "R2 values are incorrect."
    assert all(
        map(lambda rmse: pytest.approx(rmse, 1e-3) == 1.118, result["RMSE"])
    ), "RMSE values are incorrect."
    assert all(
        map(lambda mae: pytest.approx(mae, 1) == 0.8, result["MAE"])
    ), "MAE values are incorrect."

def test_sklearn_comparison():
    class MockLinearRegression:
        """Mock class for a custom linear regression model."""

        def __init__(self, coefficient, intercept):
            self.coefficients = coefficient
            self.intercept = intercept

    def mock_linreg():
        """Fixture to create a mock linear regression model."""
        return MockLinearRegression(coefficient=2.0, intercept=1.0)
    
    # Generate simple test data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

    # Run the function
    result = sklearn_comparison(x, y, mock_linreg())

    # Check if result is a dictionary
    assert isinstance(result, dict), "Function should return a dictionary."

    # Check if keys exist in the returned dictionary
    assert "custom_coefficient" in result, "Missing 'custom_coefficient' key."
    assert "custom_intercept" in result, "Missing 'custom_intercept' key."
    assert "sklearn_coefficient" in result, "Missing 'sklearn_coefficient' key."
    assert "sklearn_intercept" in result, "Missing 'sklearn_intercept' key."

    # Validate coefficient and intercept values
    expected_sklearn_model = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    expected_sklearn_model.fit(x_reshaped, y)

    assert result["custom_coefficient"] == pytest.approx(
        2.0, 1e-6
    ), "Custom coefficient is incorrect."
    assert result["custom_intercept"] == pytest.approx(
        1.0, 1e-6
    ), "Custom intercept is incorrect."
    assert result["sklearn_coefficient"] == pytest.approx(
        expected_sklearn_model.coef_[0], 1e-6
    ), "Sklearn coefficient is incorrect."
    assert result["sklearn_intercept"] == pytest.approx(
        expected_sklearn_model.intercept_, 1e-6
    ), "Sklearn intercept is incorrect."