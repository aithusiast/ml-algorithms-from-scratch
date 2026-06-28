import numpy as np
from typing import Optional
from collections import Counter


class DecisionTreeClassifier:
    def __init__(
            self,
            criterion: str,
            max_depth: int,
            min_samples_split: int,
            min_samples_leaf: int
    ):
        if criterion not in ['gini', 'entropy']:
            raise ValueError(f"Criterion must be 'gini' or 'entropy', got {criterion}!")
        
        names = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        for num, name in zip([max_depth, min_samples_split, min_samples_leaf], names):
            if num <= 0:
                raise ValueError(f"{name} must be >0, got {num}!")
            
        self.criterion= criterion
        self.max_depth= max_depth
        self.min_samples_split= min_samples_split
        self.min_samples_leaf= min_samples_leaf

    # ---- Public Methods ------------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        pass

    def predict_proba(self, X: np.ndarray):
        pass

    def get_rapport(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
    
    # ---- Validation Methods ------------------------------------------------------------------------
    def _validate_train_data(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        
        # checking for shapes and dimensions
        if X.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D array!")
        elif X.ndim == 1:
            X = X.reshape(-1,1)
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D arrat!")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X contains {X.shape[0]} samples, but y only has {y.shape[0]} samples!")
        
        # checking for target variable
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"y must only contains [0, 1], got {unique}!")
        
        # checking for dtypes
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"X must contains numeric data, got {X.dtype}!")
        
        return X, y

    def _validate_predict_input(self):
        pass
    
    # ---- Internal Methods ------------------------------------------------------------------------
    def _compute_gini(self, r_counts: Counter, l_counts: Counter):
        """
        This function take of the right, and left leaves, then it computes the weighted
        gini impurity for the split.
        """
        # counts of the right leaf
        r_no, r_yes = r_counts[0], r_counts[1]
        r_total = r_counts.total()
        
        # counts of the left leaf
        l_no, l_yes = l_counts[0], l_counts[1]
        l_total = l_counts.total()
        
        # total of samples
        total = r_total + l_total
        
        # gini impurity for each leaf
        r_gi = 1 - (r_yes/r_total)**2 - (r_no/r_total)**2
        l_gi = 1 - (l_yes/l_total)**2 - (l_no/l_total)**2

        return r_gi * (r_total/total) + l_gi * (l_total/total)
    
    def _split(self, features: np.ndarray, target: np.ndarray) -> dict:
        """
        This function the best feature to split on.
        """
        # Initialize parameters
        best_gini = np.inf
        f_index = None
        best_threshold = None
        r_leaf = None
        l_leaf = None
        
        # Splitting loop
        for idx, feature in enumerate(features):
            # for binary features
            if np.array_equal(np.unique(feature), [0,1]):
                threshold = 0.5
                r_leaf = target[(feature == 1)]
                l_leaf = target[(feature == 0)]
                gi = self._compute_gini(Counter(r_leaf), Counter(l_leaf))
            
            # for non binary features
            else:
                # computing the thresholds
                sorted_f = np.sort(feature)
                means = np.unique((sorted_f[:-1] + sorted_f[1:])/2)
                gini_impurity: list[float] = []

                # determining the best threshold
                for mean in means:
                    r_leaf = target[(feature >= mean)]
                    l_leaf = target[(feature < mean)]
                    gini_impurity.append(self._compute_gini(Counter(r_leaf), Counter(l_leaf)))
                min_index = np.argmin(gini_impurity)
                gi = gini_impurity[min_index]
                threshold = means[min_index]

            # adjusting the parameters to the optimal values
            if gi <= best_gini:
                best_gini = gi
                f_index = idx
                best_threshold = threshold
                right_leaf = r_leaf
                left_leaf = l_leaf

        return {
            'feature_idx': f_index,
            'gini_impurity': best_gini,
            'threshold': float(best_threshold),
            'r_leaf': right_leaf,
            'l_leaf': left_leaf
        }



class DecisionTreeRegressor:
    def __init__(
            self,
            criterion: str,
            max_depth: int,
            min_samples_split: int,
            min_samples_leaf: int
    ):
        if criterion not in ['gini', 'entropy']:
            raise ValueError(f"Criterion must be 'gini' or 'entropy', got {criterion}!")
        
        names = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        for num, name in zip([max_depth, min_samples_split, min_samples_leaf], names):
            if num <= 0:
                raise ValueError(f"{name} must be >0, got {num}!")
            
        self.criterion= criterion
        self.max_depth= max_depth
        self.min_samples_split= min_samples_split
        self.min_samples_leaf= min_samples_leaf
     
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def get_rapport(self):
        pass

    def _validate_train_data(self):
        pass

    def _validate_predict_input(self):
        pass