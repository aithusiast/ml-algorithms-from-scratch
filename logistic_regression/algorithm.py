import numpy as np
from typing import Optional

class LogisticRegression:
    
    def __init__(self):
        self.coef_: Optional[np.ndarray] = None # Initial values of weights
        self.intercept_: float = 0.0 # Initial value of bias

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            lr= 0.1, # learning rate
            epochs= 1000 # number of iterations
    ) -> dict:
        """
        The model training
        """
        # validate the input
        X_train, y_train = self._validate_input(X_train, y_train)

        # the data shape
        self._m, self._n = X_train.shape

        # initializing the weight values
        self.coef_ = np.zeros(self._n)

        # training the model
        return self._gradient_descent(X_train, y_train, lr, epochs)

    def predict(self, X: np.ndarray):
        """ This function predicts the class which the new input belongs to """
        return self._sigmoid(X)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        This function computes the sigmoid function
        """
        z = X @ self.coef_ + self.intercept_

        return 1 / (1 + np.exp(-z))
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        This function checks if the input is valid
        """
        # Checking for shapes
        if X.ndim == 1:
            X = X.reshape(-1,1)

        if y.ndim != 1:
            raise ValueError("The target feature must be a 1D array!")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}!")
        
        # Checking for dtypes
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X or y dtype is not a number!")
        
        return X, y
        
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        This function computes the cost value 
        """
        predictions = self._sigmoid(X)
        cost = np.sum(y * np.log1p(predictions) + (1 - y) * np.log1p(1 - predictions)) / self._m

        return float(cost)

    def _derivatives(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """ 
        This function computes the partial derivatives of weights and bias 
        """
        predictions = self._sigmoid(X)
        errors = predictions - y

        dj_dw = (X.T @ errors) / self._m
        dj_db = float(errors.sum() / self._m)

        return dj_dw, dj_db

    def _gradient_descent(
            self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float,
            epochs: int
    ) -> dict:
        """
        This function implements the gradient descent algorithm
        """
        
        j_history: list[float] = []

        # Training loop
        for i in range(epochs):
            
            # Cost value
            cost = self._compute_cost(X, y)

            if np.isnan(cost) or np.isinf(cost):
                break

            # History
            j_history.append(cost)

            # Convergence check
            if i > 0 and abs(j_history[-2] - j_history[-1]) < 1e-6:
                break
          
            # Partial derivatives
            dj_dw, dj_db = self._derivatives(X, y)
            
            # Updating weight and bias values
            self.coef_ -= lr * dj_dw
            self.intercept_ -= lr * dj_db

        return  {
            "cost_history": j_history,
            "iterations": i + 1
        }
    