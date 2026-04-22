from typing import Optional
import numpy as np

class LinearRegression:
    def __init__(self):
        self.w: Optional[np.ndarray] = None # weight
        self.b: float = 0.0 # bias

    # ── public interface ───────────────────────────────────────────────────────
    def fit(self, x: np.ndarray, y: np.ndarray,
            lr: float = 0.01, epochs: int = 1000) -> dict:
        """Train the model."""
        return self._gradient_descent(x, y, lr, epochs)

    # ── internal methods ───────────────────────────────────────────────────────

    def _cost_function(self, x: np.ndarray, y: np.ndarray, m: int) -> float:
        """
        This function computes the cost function for a given weight and bias,
        then returns the cost function value
        """
        predictions = (x @ self.w) + self.b
        errors = predictions - y
        cost = np.sum(errors ** 2) / (2 * m)

        return float(cost)

    def _compute_derivatives(self, x: np.ndarray, y: np.ndarray, m: int) -> tuple[np.ndarray, float]:
        """
        This function computes the parial derivatives of a given weight and bias,
        and return their values
        """
        predictions = (x @ self.w) + self.b
        errors = predictions - y

        dj_dw = (x.T @ errors) / m # Weight partial derivative(s)
        dj_db = float(np.sum(errors) / m) # Bias partial derivative

        return dj_dw, dj_db
    
    def _validate_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        x = np.asarray(x, dtype= float)
        y = np.asarray(y, dtype= float)

        if x.ndim == 1:
            x = x.reshape(-1,1)
        
        if y.ndim != 1:
            raise ValueError("Target must be a 1D array.")

        if x.shape[0] != y.shape[0]:
            raise ValueError( f"X has {x.shape[0]} samples but y has {y.shape[0]}.")
        
        return x, y

    def _gradient_descent(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            lr: float, # Learning rate value
            epochs: int # Number of iterations (for model training)
    ) -> dict:
        """
        This function implemnt the gradient descent algorithm
        """

        # Data validation
        x_train, y_train = self._validate_data(x_train, y_train)
        
        # Data shape
        m, n = x_train.shape 

        # Cost function values history
        j_history: list[float] = [] 

        # Initializing weight and bias
        self.w = np.zeros(n)
        self.b = 0.0
        
        # Training loop
        for i in range(epochs):
            
            # Cost value
            cost = self._cost_function(x_train, y_train, m)

            if np.isnan(cost) or np.isinf(cost):
                break

            # History
            j_history.append(cost)

            # Convergence check
            if i > 0 and abs(j_history[-2] - j_history[-1]) < 1e-6:
                break
          
            # Partial derivatives
            dj_dw, dj_db = self._compute_derivatives(x_train, y_train, m)
            
            # Updating weight and bias values
            self.w -= lr * dj_dw
            self.b -= lr * dj_db

        return  {
            "cost_history": j_history,
            "weight": self.w,
            "bias": self.b,
            "iterations": i + 1
        }