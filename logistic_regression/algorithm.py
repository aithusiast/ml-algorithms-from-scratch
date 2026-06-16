import numpy as np
from typing import Optional

class LogisticRegression:
    
    def __init__(
            self,
            lr: float= 0.1, # learning rate
            epochs: int= 1000, # number of iterations (model training)
            tol: float= 1e-6, # convergence tolerance
    ) -> None:
        
        if lr <= 0:
            raise ValueError(f"The learning rate must be positive, got {lr}")
        if epochs < 1:
            raise ValueError(f"'epochs' must be >1, got {epochs}")
        
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

        self.coef_: Optional[np.ndarray]= None
        self.intercept_: float= 0.0

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ) -> LogisticRegression:
        """
        The model training
        """
        # validate the input
        X_train, y_train = self._validate_input(X_train, y_train)

        # the data shape
        m_samples, n_features = X_train.shape

        # initializing the weights and bias
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        # cost history
        self.j_history_: list[float] = []

        # stopping iteration
        self.iter_: int= 0

        # training the model
        self._gradient_descent(X_train, y_train, m_samples)

        return self
    
    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        """
        This function validate the prediction method input
        """
        X = np.asarray(X)

        # Checking if the model is trained
        if self.coef_ is None:
            raise ValueError("Model is not trained yet. run .fit() first!")
        
        # Checking dimensions and shapes
        if X.ndim == 1:
            X = X.reshape(1,-1)

        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")
        
        if X.shape[-1] != self.coef_.shape[0]:
            raise ValueError(f"X has {X.shape[-1]} features, but the model was trained on {self.coef_.shape[0]} features!")
        
        return X
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        This function computes the probabilities of each sample
        """
        X = self._validate_predict_input(X)
        return self._sigmoid(X)

    def predict(self, X: np.ndarray, threshold: float=0.5):
        """
        This function predicts the class which the new input belongs to
        """
        # Checking threshold value
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold value must be in range (0,1), got {threshold}")
        
        return (self.predict_proba(X) >= threshold).astype(int)

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
        X = np.asarray(X)
        y = np.asarray(y)

        # Checking for dimensions and shapes
        if X.ndim == 1:
            X = X.reshape(-1,1)

        if X.ndim > 2:
            raise ValueError(f"X must be 1D or 2D array, got {X.ndim}D array!")

        if y.ndim != 1:
            raise ValueError("The target feature must be a 1D array!")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}!")
        
        # Checking for dtypes
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("X and y must contain numeric values!")
        
        # Checking for classes in target features
        uniques = np.unique(y)
        if not np.all(np.isin(uniques, [0,1])):
            raise ValueError(f"The target feature must only contains 0 and 1 classes, found {uniques}")
        
        return X, y
        
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, m_samples: int) -> float:
        """
        This function computes the cost value 
        """
        predictions = self._sigmoid(X)
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        cost = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) / m_samples

        return float(cost)

    def _derivatives(self, X: np.ndarray, y: np.ndarray, m_samples: int) -> tuple[np.ndarray, float]:
        """ 
        This function computes the partial derivatives of weights and bias 
        """
        predictions = self._sigmoid(X)
        errors = predictions - y

        dj_dw = (X.T @ errors) / m_samples
        dj_db = float(errors.sum() / m_samples)

        return dj_dw, dj_db

    def _gradient_descent(
            self,
            X: np.ndarray,
            y: np.ndarray,
            m_samples: int
    ) -> None:
        """
        This function implements the gradient descent algorithm
        """
        # Training loop
        for i in range(self.epochs):
            
            # Cost value
            cost = self._compute_cost(X, y, m_samples)

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
            dj_dw, dj_db = self._derivatives(X, y, m_samples)
            
            # Updating weights and bias
            self.coef_ -= self.lr * dj_dw
            self.intercept_ -= self.lr * dj_db
    