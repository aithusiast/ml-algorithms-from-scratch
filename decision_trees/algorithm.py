import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(
            self,
            max_depth: int,
            min_samples_split: int,
            min_samples_leaf: int
    ):  
        names = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        for num, name in zip([max_depth, min_samples_split, min_samples_leaf], names):
            if num <= 0:
                raise ValueError(f"{name} must be >0, got {num}!")
            
        self.max_depth= max_depth
        self.min_samples_split= min_samples_split
        self.min_samples_leaf= min_samples_leaf
        self.root = None

    # ---- Public Methods ------------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DecisionTreeClassifier":
        X_train, y_train = self._validate_train_data(X_train, y_train)
        self.n_features = X_train.shape[-1]
        self.root = self._build_tree(X_train, y_train, depth= 0)
        return self
    
    def predict(self, X: np.ndarray) -> list[int]:
        X = self._validate_predict_input(X)
        predictions = [self._predict_one(x) for x in X]
        return predictions

    def print_tree(self, node=None, depth=0, indent="|   "):
        prefix = indent * depth
        if node is None:
            node = self.root

        if isinstance(node, Leaf):
            print(f"{prefix}|--- class: {node.prediction}")
            return

        feature_label = f"Feature[{node.feature_idx}]"

        print(f"{prefix}|--- {feature_label} <= {node.threshold}")
        self.print_tree(node.right, depth + 1, indent)

        print(f"{prefix}|--- {feature_label} > {node.threshold}")
        self.print_tree(node.left, depth + 1, indent)
    
    # ---- Validation Methods ------------------------------------------------------------------------
    def _validate_train_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y)
        
        # checking for shapes and dimensions
        if X.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D array!")
        elif X.ndim == 1:
            X = X.reshape(-1,1)
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D array!")
        
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

    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        # checking for shapes & dimension
        if X.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D array!")
        
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        if X.shape[-1] != self.n_features:
            raise ValueError(f"X has {X.shape[-1]} features, but the model was trained on {self.n_features} features!")
        
        # checking for dtype
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contains numeric data!")
        
        return X
    
    # ---- Internal Methods ------------------------------------------------------------------------
    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while isinstance(node, Node):
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
    
    def _compute_gini(self, r_counts: Counter, l_counts: Counter) -> float:
        # total of each count
        r_total = r_counts.total()
        l_total = l_counts.total()
        
        # total of samples
        total = r_total + l_total
        
        # gini impurity for each leaf
        r_gi = self._single_gini(r_counts)
        l_gi = self._single_gini(l_counts)
        w_gi = r_gi * (r_total/total) + l_gi * (l_total/total)
        return w_gi
    
    def _split(self, X: np.ndarray, y: np.ndarray) -> dict:
        best_split = {
            "feature_idx": None,
            "gini_impurity": np.inf,
            "threshold": None,
            "r_mask": None,
            "l_mask": None
        }
        
        for idx in range(self.n_features):
            unique = np.unique(X[:,idx])
            if len(unique) < 2:
                continue

            thresholds = (unique[1:] + unique[:-1])/2
            for threshold in thresholds:
                r_mask = X[:,idx] >= threshold
                l_mask = ~r_mask
                if r_mask.sum() < self.min_samples_leaf or l_mask.sum() < self.min_samples_leaf:
                    continue
                w_gi= self._compute_gini(Counter(y[r_mask]), Counter(y[l_mask]))
                if w_gi <= best_split["gini_impurity"]:
                    best_split.update({
                        "feature_idx": idx,
                        "gini_impurity": w_gi,
                        "threshold": threshold,
                        "r_mask": r_mask,
                        "l_mask": l_mask,
                    })

        return best_split
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int= 0) -> 'Node | Leaf':
        if depth >= self.max_depth:
            return Leaf(prediction= self._majority_class(y))
        
        if len(y) < self.min_samples_split:
            return Leaf(prediction= self._majority_class(y))
        
        if len(np.unique(y)) == 1:
            return Leaf(prediction= y[0])
        
        best_split = self._split(X, y)

        if best_split["feature_idx"] is None:
            return Leaf(prediction=self._majority_class(y))

        l_mask = best_split['l_mask']
        r_mask = best_split['r_mask']
        
        right = self._build_tree(X[r_mask], y[r_mask], depth+1)
        left = self._build_tree(X[l_mask], y[l_mask], depth+1)
    
        return Node(
            feature_idx= best_split['feature_idx'],
            threshold= best_split['threshold'],
            left= left,
            right= right
        )
    
    @staticmethod
    def _single_gini(counts: Counter) -> float:
        total = counts.total()
        if total == 0:
            return 0
        return 1 - sum((c/total)**2 for c in counts.values())
    
    @staticmethod
    def _majority_class(y: Counter) -> int:
        return Counter(y).most_common(1)[0][0]

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, prediction):
        self.prediction = prediction
