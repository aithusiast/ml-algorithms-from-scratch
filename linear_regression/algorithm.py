from typing import Optional
import numpy as np

class LinearRegression:
    def __init__(self):
        self.w_: Optional[np.ndarray] = None # weight
        self.b_: float = 0.0 # bias

    # ── public interface ───────────────────────────────────────────────────────
    def fit(self, x: np.ndarray, y: np.ndarray,
            lr: float = 0.01, epochs: int = 1000) -> dict:
        """Train the model."""
        # validate the input
        x, y = self._validate_input(x, y)

        # the data shape
        self._m, self._n = x.shape

        # initializing the weight values
        self.w_ = np.zeros(self._n)

        # training the model
        return self._gradient_descent(x, y, lr, epochs)

    # ── internal methods ───────────────────────────────────────────────────────

    def _cost_function(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function computes the cost function for a given weight and bias,
        then returns the cost function value
        """
        predictions = (x @ self.w_) + self.b_
        errors = predictions - y
        cost = np.sum(errors ** 2) / (2 *self._m)

        return float(cost)

    def _compute_derivatives(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        This function computes the parial derivatives of a given weight and bias,
        and return their values
        """
        predictions = (x @ self.w_) + self.b_
        errors = predictions - y

        dj_dw = (x.T @ errors) / self._m # Weight partial derivative(s)
        dj_db = float(np.sum(errors) / self._m) # Bias partial derivative

        return dj_dw, dj_db
    
    def _validate_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This function checks if the input is valid
        """
        # Checking for shapes
        if x.ndim == 1:
            x = x.reshape(-1,1)
        
        if y.ndim != 1:
            raise ValueError("Target must be a 1D array!")

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"X has {x.shape[0]} samples but y has {y.shape[0]}!")
        
        # Checking for dtypes
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X or y dtype is not a number!")
        
        return x, y

    def _gradient_descent(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            lr: float, # Learning rate value
            epochs: int # Number of iterations (for model training)
    ) -> dict:
        """
        This function implement the gradient descent algorithm
        """
        # Cost function values history
        j_history: list[float] = [] 
        
        # Training loop
        for i in range(epochs):
            
            # Cost value
            cost = self._cost_function(x_train, y_train)

            if np.isnan(cost) or np.isinf(cost):
                break

            # History
            j_history.append(cost)

            # Convergence check
            if i > 0 and abs(j_history[-2] - j_history[-1]) < 1e-6:
                break
          
            # Partial derivatives
            dj_dw, dj_db = self._compute_derivatives(x_train, y_train)
            
            # Updating weight and bias values
            self.w -= lr * dj_dw
            self.b -= lr * dj_db

        return  {
            "cost_history": j_history,
            "iterations": i + 1
        }
    