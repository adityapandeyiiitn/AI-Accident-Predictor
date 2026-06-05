import numpy as np
import pandas as pd

class CustomPreprocessor:
    """
    A custom scikit-learn style preprocessor that One-Hot Encodes categorical features
    and passes numerical features through.
    """
    def __init__(self, categorical_cols, numerical_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.categories_ = {}
        self.feature_names_out_ = []

    def fit(self, df):
        self.categories_ = {}
        self.feature_names_out_ = []
        # Fit categorical columns (saving unique categories in sorted order)
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].unique())
            self.categories_[col] = unique_vals
            for val in unique_vals:
                self.feature_names_out_.append(f"{col}_{val}")
        # Fit numerical columns
        for col in self.numerical_cols:
            self.feature_names_out_.append(col)
        return self

    def transform(self, df):
        encoded_parts = []
        for col in self.categorical_cols:
            cats = self.categories_.get(col, [])
            for val in cats:
                encoded_parts.append((df[col] == val).astype(float).values)
        for col in self.numerical_cols:
            encoded_parts.append(df[col].astype(float).values)
        return np.column_stack(encoded_parts)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_feature_names_out(self):
        return self.feature_names_out_


class SimpleDecisionTreeRegressor:
    """
    A custom pure-Python/NumPy Decision Tree Regressor.
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        self.tree = self._build_tree(X, y, depth=0)
        # Normalize feature importances
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.std(y) == 0):
            return {'val': np.mean(y)}

        # Feature selection (Random Forest feature subsetting)
        features = np.arange(n_features)
        if self.max_features == 'sqrt':
            n_sub = int(np.sqrt(n_features))
            features = np.random.choice(features, size=max(1, n_sub), replace=False)
        elif isinstance(self.max_features, int):
            features = np.random.choice(features, size=min(self.max_features, n_features), replace=False)

        best_feat, best_thresh = None, None
        best_mse = float('inf')
        current_variance = np.var(y) * n_samples

        for feat in features:
            thresholds = np.unique(X[:, feat])
            # Speed up continuous threshold search by binning if there are many values
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 10))
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                left_var = np.var(y[left_idx]) * np.sum(left_idx)
                right_var = np.var(y[right_idx]) * np.sum(right_idx)
                mse = left_var + right_var
                
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None or (current_variance - best_mse) <= 0:
            return {'val': np.mean(y)}

        # Update importance using variance reduction (impurity gain)
        importance_gain = (current_variance - best_mse) / len(y)
        self.feature_importances_[best_feat] += importance_gain

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            'feat': best_feat,
            'thresh': best_thresh,
            'left': left_child,
            'right': right_child
        }

    def predict(self, X):
        return np.array([self._predict_row(self.tree, row) for row in X])

    def _predict_row(self, node, row):
        if 'val' in node:
            return node['val']
        if row[int(node['feat'])] <= node['thresh']:
            return self._predict_row(node['left'], row)
        return self._predict_row(node['right'], row)


class SimpleRandomForestRegressor:
    """
    A custom pure-Python/NumPy Random Forest Regressor.
    """
    def __init__(self, n_estimators=15, max_depth=6, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            tree = SimpleDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Aggregate feature importances
            importances += tree.feature_importances_
            
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


class SimpleLinearRegression:
    """
    A custom Ridge (L2 regularized OLS) Linear Regression model in NumPy.
    """
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Column of ones for intercept
        X_design = np.column_stack([np.ones(n_samples), X])
        
        # Closed form Ridge Regression solution: (X^T X + alpha * I)^(-1) X^T y
        n_features_design = X_design.shape[1]
        I = np.eye(n_features_design)
        I[0, 0] = 0.0  # Do not regularize the intercept term
        
        A = X_design.T @ X_design + self.alpha * I
        b = X_design.T @ y
        
        beta = np.linalg.solve(A, b)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class CustomPipeline:
    """
    A custom scikit-learn style Pipeline containing a preprocessor and regressor.
    """
    def __init__(self, preprocessor, regressor):
        self.preprocessor = preprocessor
        self.regressor = regressor

    def fit(self, X_df, y):
        X_trans = self.preprocessor.fit_transform(X_df)
        self.regressor.fit(X_trans, y)
        return self

    def predict(self, X_df):
        X_trans = self.preprocessor.transform(X_df)
        return self.regressor.predict(X_trans)
