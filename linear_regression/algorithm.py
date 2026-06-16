from typing import Optional
import numpy as np

class LinearRegression:
    def __init__(
            self,
            lr: float= 0.1, # learning rate
            epochs: int= 1000, # number of iterations (model training)
            tol: float= 1e-7 # convergence tolerance
    ) -> None:
        
        # Checking parameters
        if lr <= 0:
            raise ValueError(f"The learning rate must be positive, got {lr}!")
        
        if epochs < 1:
            raise ValueError(f"'epochs' must be >1, got {epochs}!")
        
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

        self.coef_: Optional[np.ndarray] = None # weights
        self.intercept_: float = 0.0 # bias

    # ── public interface ───────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Train the model."""
        # validate the input
        X, y = self._validate_data(X, y)

        # the data shape
        m_samples, n_features = X.shape

        # initializing the weights and bias
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        # cost history, and stopping iteration
        self.j_history_: list[float]= []
        self.iter_: int= 0

        # training the model
        self._gradient_descent(X, y, m_samples)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_predict_input(X)
        return X @ self.coef_ + self.intercept_

    
    # ── data validation ───────────────────────────────────────────────────────

    def _validate_data(self, X: np.ndarray, y: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        """
        This function checks if the input is valid
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Checking for shapes
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        if y.ndim != 1:
            raise ValueError("Target must be a 1D array!")
        
        if X.ndim > 2:
            raise ValueError(f"X must be a 1D or 2D array, got a {X.ndim}D array!")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}!")
        
        # Checking for dtypes
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X or y dtype is not a number!")
        
        return X, y
    
    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        """
        This function validates the input of the the predict function
        """
        X = np.asarray(X)

        # Checking if model is trained
        if self.coef_ is None:
            raise ValueError("Model is not trained yet, run .fit() first!")
        
        # Checking for shapes and dimensions
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")
        
        if X.shape[-1] != self.coef_.shape[0]:
            raise ValueError(f"X has {X.shape[-1]} features, but the model was trained on {self.coef_.shape[0]} features!")
        
        # Checking for dtype
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contains numeric values!")
        
        return X

    # ── internal methods ───────────────────────────────────────────────────────

    def _cost_function(self, x: np.ndarray, y: np.ndarray, m_samples: int) -> float:
        """
        This function computes the cost function for a given weight and bias,
        then returns the cost function value
        """
        predictions = (x @ self.coef_) + self.intercept_
        errors = predictions - y
        cost = np.sum(errors ** 2) / (2 *m_samples)

        return float(cost)

    def _compute_derivatives(self, X: np.ndarray, y: np.ndarray, m_samples) -> tuple[np.ndarray, float]:
        """
        This function computes the parial derivatives of a given weight and bias,
        and return their values
        """
        predictions = (X @ self.coef_) + self.intercept_
        errors = predictions - y

        dj_dw = (X.T @ errors) / m_samples # Weight partial derivative(s)
        dj_db = float(np.sum(errors) / m_samples) # Bias partial derivative

        return dj_dw, dj_db

    def _gradient_descent(self, X_train: np.ndarray, y_train: np.ndarray, m_samples: int) -> None:
        """
        This function implement the gradient descent algorithm
        """
        # Training loop
        for i in range(self.epochs):
            
            # Cost value
            cost = self._cost_function(X_train, y_train, m_samples)

            if np.isnan(cost) or np.isinf(cost):
                raise RuntimeError(
                    f"Training diverged at epoch {i}, (cost= {cost}). \nTry a smaller learning rate or normalize your features!"
                )

            # History and stopping iteration
            self.j_history_.append(cost)
            self.iter_ = i + 1

            # Convergence check
            if i > 0 and abs(self.j_history_[-2] - self.j_history_[-1]) < self.tol:
                break
          
            # Partial derivatives
            dj_dw, dj_db = self._compute_derivatives(X_train, y_train, m_samples)
            
            # Updating weight and bias values
            self.coef_ -= self.lr * dj_dw
            self.intercept_ -= self.lr * dj_db
