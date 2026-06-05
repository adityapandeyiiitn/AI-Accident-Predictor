import os
import nbformat as nbf

# Define the custom classes source code as a string to inject into the notebook
custom_classes_source = """# ==============================================================================
# CUSTOM MACHINE LEARNING COMPONENTS (Pure NumPy/Pandas)
# Written to bypass OS-level DLL policies that block C-compiled extensions (like scikit-learn).
# ==============================================================================

class CustomPreprocessor:
    \"\"\"
    A custom scikit-learn style preprocessor that One-Hot Encodes categorical features
    and passes numerical features through.
    \"\"\"
    def __init__(self, categorical_cols, numerical_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.categories_ = {}
        self.feature_names_out_ = []

    def fit(self, df):
        self.categories_ = {}
        self.feature_names_out_ = []
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].unique())
            self.categories_[col] = unique_vals
            for val in unique_vals:
                self.feature_names_out_.append(f"{col}_{val}")
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
    \"\"\"
    A custom pure-Python/NumPy Decision Tree Regressor.
    \"\"\"
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
    \"\"\"
    A custom pure-Python/NumPy Random Forest Regressor.
    \"\"\"
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
            importances += tree.feature_importances_
            
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


class SimpleLinearRegression:
    \"\"\"
    A custom Ridge (L2 regularized OLS) Linear Regression model in NumPy.
    \"\"\"
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        X_design = np.column_stack([np.ones(n_samples), X])
        n_features_design = X_design.shape[1]
        I = np.eye(n_features_design)
        I[0, 0] = 0.0
        A = X_design.T @ X_design + self.alpha * I
        b = X_design.T @ y
        beta = np.linalg.solve(A, b)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class CustomPipeline:
    \"\"\"
    A custom scikit-learn style Pipeline containing a preprocessor and regressor.
    \"\"\"
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
"""

def create_jupyter_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Accident Risk Zone Predictor - Model Training & EDA\n"
        "This Jupyter Notebook contains the complete Exploratory Data Analysis (EDA), Preprocessing, "
        "and Machine Learning Model training for the **Accident Risk Zone Predictor** project.\n\n"
        "### Project Objectives:\n"
        "1. Perform Exploratory Data Analysis to identify factors contributing to high accident risk.\n"
        "2. Build custom Preprocessing pipelines using pure NumPy/Pandas (`CustomPreprocessor`).\n"
        "3. Train and compare two custom models from scratch: **Random Forest Regressor** and **Linear Regression**.\n"
        "4. Evaluate performance using RMSE and $R^2$ scores.\n"
        "5. Export the best model for real-time predictions in the Streamlit app."
    ))
    
    # Cell 2: Code (Imports)
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n\n"
        "# Set style for plotting\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "print(\"Libraries successfully imported!\")"
    ))
    
    # Cell 3: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## Custom Machine Learning Components\n"
        "Below are our custom preprocessor and model classes written in pure Python & NumPy "
        "to ensure 100% execution compatibility, bypassing binary/compiled C-extension DLL policy blocks."
    ))
    
    # Cell 4: Code (Custom Classes)
    cells.append(nbf.v4.new_code_cell(custom_classes_source))
    
    # Cell 5: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Dataset\n"
        "We load the generated synthetic dataset (`dataset.csv`) and inspect its columns and values."
    ))
    
    # Cell 6: Code
    cells.append(nbf.v4.new_code_cell(
        "df = pd.read_csv('dataset.csv')\n"
        "print(f\"Dataset Shape: {df.shape}\")\n"
        "df.head()"
    ))
    
    # Cell 7: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n"
        "Let's examine data types, check for missing values, and analyze statistical summaries."
    ))
    
    # Cell 8: Code
    cells.append(nbf.v4.new_code_cell(
        "df.info()"
    ))
    
    # Cell 9: Code
    cells.append(nbf.v4.new_code_cell(
        "df.describe(include='all')"
    ))
    
    # Cell 10: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2.1 Distribution of Accident Risk Score\n"
        "Let's visualize the target variable `accident_risk` to see the spread and identify peaks."
    ))
    
    # Cell 11: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(df['accident_risk'], kde=True, color='crimson', bins=30)\n"
        "plt.title('Distribution of Accident Risk Scores')\n"
        "plt.xlabel('Accident Risk Score (0 - 100)')\n"
        "plt.ylabel('Frequency')\n"
        "plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 12: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2.2 Accident Risk vs Hour of Day\n"
        "Analyze how the average accident risk score fluctuates across 24 hours. "
        "We expect risk spikes during rush hours and late-night periods."
    ))
    
    # Cell 13: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(12, 6))\n"
        "sns.lineplot(data=df, x='hour', y='accident_risk', marker='o', color='purple', errorbar=None)\n"
        "plt.title('Average Accident Risk Score by Hour of Day')\n"
        "plt.xlabel('Hour (0 - 23)')\n"
        "plt.ylabel('Average Accident Risk')\n"
        "plt.xticks(range(0, 24))\n"
        "plt.savefig('risk_vs_hour.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 14: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2.3 Accident Risk by Weather Conditions\n"
        "We'll examine risk values under Clear, Rain, and Fog weather conditions."
    ))
    
    # Cell 15: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 5))\n"
        "sns.barplot(data=df, x='weather', y='accident_risk', palette='coolwarm', hue='weather', legend=False, errorbar=None)\n"
        "plt.title('Average Accident Risk by Weather Condition')\n"
        "plt.xlabel('Weather')\n"
        "plt.ylabel('Average Accident Risk')\n"
        "plt.savefig('risk_vs_weather.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 16: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2.4 Accident Risk by Traffic Density\n"
        "Checking risk levels against Low, Medium, and High traffic densities."
    ))
    
    # Cell 17: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 5))\n"
        "sns.boxplot(data=df, x='traffic', y='accident_risk', palette='viridis', hue='traffic', legend=False)\n"
        "plt.title('Accident Risk Distribution by Traffic Level')\n"
        "plt.xlabel('Traffic Level')\n"
        "plt.ylabel('Accident Risk Score')\n"
        "plt.savefig('risk_vs_traffic.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 18: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 2.5 Compound Insights: Weather + Area Type Interaction\n"
        "Let's see how risk levels compound when combining area type (urban, highway, rural) "
        "with bad weather. For example, highway driving in fog is simulated to be a peak risk event."
    ))
    
    # Cell 19: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(10, 6))\n"
        "sns.barplot(data=df, x='area_type', y='accident_risk', hue='weather', palette='magma', errorbar=None)\n"
        "plt.title('Accident Risk: Area Type vs Weather Interaction')\n"
        "plt.xlabel('Area Type')\n"
        "plt.ylabel('Average Accident Risk')\n"
        "plt.savefig('risk_area_weather_interaction.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 20: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Preprocessing & Correlation Analysis\n"
        "We'll perform a quick temporary mapping of categories to plot a correlation heatmap "
        "to confirm that simulated patterns were correctly encoded."
    ))
    
    # Cell 21: Code
    cells.append(nbf.v4.new_code_cell(
        "df_encoded = df.copy()\n"
        "df_encoded['weather'] = df_encoded['weather'].map({'clear': 0, 'rain': 1, 'fog': 2})\n"
        "df_encoded['traffic'] = df_encoded['traffic'].map({'low': 0, 'medium': 1, 'high': 2})\n"
        "df_encoded['area_type'] = df_encoded['area_type'].map({'rural': 0, 'urban': 1, 'highway': 2})\n\n"
        "plt.figure(figsize=(8, 6))\n"
        "sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n"
        "plt.title('Feature Correlation Heatmap')\n"
        "plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 22: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Machine Learning Modeling\n"
        "We split the dataset into features ($X$) and target ($y$), perform an 80/20 train/test split, "
        "and configure our preprocessing pipelines using our custom preprocessing tools."
    ))
    
    # Cell 23: Code
    cells.append(nbf.v4.new_code_cell(
        "X = df.drop(columns=['accident_risk'])\n"
        "y = df['accident_risk']\n\n"
        "# Simple manual train/test split to avoid dependency\n"
        "np.random.seed(42)\n"
        "shuffled_indices = np.random.permutation(len(df))\n"
        "test_set_size = int(len(df) * 0.2)\n"
        "test_indices = shuffled_indices[:test_set_size]\n"
        "train_indices = shuffled_indices[test_set_size:]\n\n"
        "X_train, X_test = X.iloc[train_indices].reset_index(drop=True), X.iloc[test_indices].reset_index(drop=True)\n"
        "y_train, y_test = y.iloc[train_indices].reset_index(drop=True).values, y.iloc[test_indices].reset_index(drop=True).values\n\n"
        "print(f\"Training set shape: {X_train.shape}\")\n"
        "print(f\"Testing set shape: {X_test.shape}\")"
    ))
    
    # Cell 24: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 4.1 Define the Preprocessor\n"
        "Using `CustomPreprocessor` to encode categorical columns while passing numerical values through."
    ))
    
    # Cell 25: Code
    cells.append(nbf.v4.new_code_cell(
        "categorical_cols = ['weather', 'traffic', 'area_type']\n"
        "numerical_cols = ['hour']\n\n"
        "preprocessor = CustomPreprocessor(categorical_cols, numerical_cols)\n"
        "print(\"Custom Preprocessor defined successfully!\")"
    ))
    
    # Cell 26: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 4.2 Model 1: Random Forest Regressor Pipeline\n"
        "Define the Random Forest pipeline, train it on the training data, and generate predictions."
    ))
    
    # Cell 27: Code
    cells.append(nbf.v4.new_code_cell(
        "rf_pipeline = CustomPipeline(preprocessor, SimpleRandomForestRegressor(n_estimators=15, max_depth=6, max_features='sqrt'))\n"
        "rf_pipeline.fit(X_train, y_train)\n"
        "rf_preds = rf_pipeline.predict(X_test)\n"
        "print(\"Custom Random Forest Regressor trained!\")"
    ))
    
    # Cell 28: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 4.3 Model 2: Linear Regression Pipeline\n"
        "Define the baseline Linear Regression pipeline, train it, and generate predictions."
    ))
    
    # Cell 29: Code
    cells.append(nbf.v4.new_code_cell(
        "lr_pipeline = CustomPipeline(preprocessor, SimpleLinearRegression(alpha=1e-3))\n"
        "lr_pipeline.fit(X_train, y_train)\n"
        "lr_preds = lr_pipeline.predict(X_test)\n"
        "print(\"Custom Linear Regression trained!\")"
    ))
    
    # Cell 30: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Model Evaluation & Comparison\n"
        "Let's calculate RMSE (Root Mean Squared Error) and $R^2$ Score to compare both models."
    ))
    
    # Cell 31: Code
    cells.append(nbf.v4.new_code_cell(
        "def rmse(y_true, y_pred):\n"
        "    return np.sqrt(np.mean((y_true - y_pred)**2))\n\n"
        "def r2_score(y_true, y_pred):\n"
        "    ss_res = np.sum((y_true - y_pred)**2)\n"
        "    ss_tot = np.sum((y_true - np.mean(y_true))**2)\n"
        "    return 1 - (ss_res / (ss_tot + 1e-15))\n\n"
        "rf_rmse_val = rmse(y_test, rf_preds)\n"
        "rf_r2_val = r2_score(y_test, rf_preds)\n\n"
        "lr_rmse_val = rmse(y_test, lr_preds)\n"
        "lr_r2_val = r2_score(y_test, lr_preds)\n\n"
        "metrics_df = pd.DataFrame({\n"
        "    'Model': ['Random Forest Regressor', 'Linear Regression'],\n"
        "    'RMSE': [rf_rmse_val, lr_rmse_val],\n"
        "    'R² Score': [rf_r2_val, lr_r2_val]\n"
        "})\n"
        "print(metrics_df.to_string(index=False))"
    ))
    
    # Cell 32: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 5.1 Visualize Predictions vs Actual Risk (Random Forest)\n"
        "Let's see how close predicted risk scores are to actual simulated risk scores."
    ))
    
    # Cell 33: Code
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 8))\n"
        "plt.scatter(y_test, rf_preds, alpha=0.5, color='teal')\n"
        "plt.plot([0, 100], [0, 100], '--', color='red', linewidth=2)\n"
        "plt.xlabel('Actual Risk Score')\n"
        "plt.ylabel('Predicted Risk Score')\n"
        "plt.title('Random Forest: Actual vs Predicted Risk')\n"
        "plt.xlim(0, 100)\n"
        "plt.ylim(0, 100)\n"
        "plt.savefig('rf_actual_vs_predicted.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()"
    ))
    
    # Cell 34: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "### 5.2 Feature Importance Analysis\n"
        "Extract the Random Forest Regressor's feature importances to identify key risk drivers."
    ))
    
    # Cell 35: Code
    cells.append(nbf.v4.new_code_cell(
        "all_features = rf_pipeline.preprocessor.get_feature_names_out()\n"
        "importances = rf_pipeline.regressor.feature_importances_\n\n"
        "feature_importance_df = pd.DataFrame({\n"
        "    'Feature': all_features,\n"
        "    'Importance': importances\n"
        "}).sort_values(by='Importance', ascending=False)\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)\n"
        "plt.title('Feature Importances (Random Forest Regressor)')\n"
        "plt.xlabel('Importance Score')\n"
        "plt.ylabel('Features')\n"
        "plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "print(\"Top 3 Risk Factors:\")\n"
        "for idx, row in feature_importance_df.head(3).iterrows():\n"
        "    print(f\"- {row['Feature']}: {row['Importance']:.4f}\")"
    ))
    
    # Cell 36: Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Save the Best Model\n"
        "Export the Random Forest Regressor pipeline to `model.pkl` using joblib."
    ))
    
    # Cell 37: Code
    cells.append(nbf.v4.new_code_cell(
        "joblib.dump(rf_pipeline, 'model.pkl')\n"
        "print(\"Best model pipeline (Random Forest) saved successfully as 'model.pkl'.\")"
    ))
    
    nb['cells'] = cells
    
    with open('model.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("Notebook 'model.ipynb' generated successfully.")

# Run model training directly in Python to create output artifacts
def run_model_training():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    
    # Import custom components from our local ml_components module
    from ml_components import CustomPreprocessor, SimpleRandomForestRegressor, SimpleLinearRegression, CustomPipeline
    
    df = pd.read_csv('dataset.csv')
    
    # Plots styling
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['accident_risk'], kde=True, color='crimson', bins=30)
    plt.title('Distribution of Accident Risk Scores')
    plt.xlabel('Accident Risk Score (0 - 100)')
    plt.ylabel('Frequency')
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Risk vs Hour
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='hour', y='accident_risk', marker='o', color='purple', errorbar=None)
    plt.title('Average Accident Risk Score by Hour of Day')
    plt.xlabel('Hour (0 - 23)')
    plt.ylabel('Average Accident Risk')
    plt.xticks(range(0, 24))
    plt.savefig('risk_vs_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Weather
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='weather', y='accident_risk', palette='coolwarm', hue='weather', legend=False, errorbar=None)
    plt.title('Average Accident Risk by Weather Condition')
    plt.xlabel('Weather')
    plt.ylabel('Average Accident Risk')
    plt.savefig('risk_vs_weather.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Traffic
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='traffic', y='accident_risk', palette='viridis', hue='traffic', legend=False)
    plt.title('Accident Risk Distribution by Traffic Level')
    plt.xlabel('Traffic Level')
    plt.ylabel('Accident Risk Score')
    plt.savefig('risk_vs_traffic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Interaction
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='area_type', y='accident_risk', hue='weather', palette='magma', errorbar=None)
    plt.title('Accident Risk: Area Type vs Weather Interaction')
    plt.xlabel('Area Type')
    plt.ylabel('Average Accident Risk')
    plt.savefig('risk_area_weather_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation Heatmap
    df_encoded = df.copy()
    df_encoded['weather'] = df_encoded['weather'].map({'clear': 0, 'rain': 1, 'fog': 2})
    df_encoded['traffic'] = df_encoded['traffic'].map({'low': 0, 'medium': 1, 'high': 2})
    df_encoded['area_type'] = df_encoded['area_type'].map({'rural': 0, 'urban': 1, 'highway': 2})
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Split features/target
    X = df.drop(columns=['accident_risk'])
    y = df['accident_risk']
    
    # Manual 80/20 shuffle split for train/test
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    X_train, X_test = X.iloc[train_indices].reset_index(drop=True), X.iloc[test_indices].reset_index(drop=True)
    y_train, y_test = y.iloc[train_indices].reset_index(drop=True).values, y.iloc[test_indices].reset_index(drop=True).values
    
    # Define and fit pipelines
    categorical_cols = ['weather', 'traffic', 'area_type']
    numerical_cols = ['hour']
    
    preprocessor = CustomPreprocessor(categorical_cols, numerical_cols)
    
    rf_pipeline = CustomPipeline(preprocessor, SimpleRandomForestRegressor(n_estimators=15, max_depth=6, max_features='sqrt'))
    lr_pipeline = CustomPipeline(preprocessor, SimpleLinearRegression(alpha=1e-3))
    
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)
    
    rf_preds = rf_pipeline.predict(X_test)
    lr_preds = lr_pipeline.predict(X_test)
    
    # Evaluation functions
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
        
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / (ss_tot + 1e-15))
        
    rf_rmse_val = rmse(y_test, rf_preds)
    rf_r2_val = r2_score(y_test, rf_preds)
    
    lr_rmse_val = rmse(y_test, lr_preds)
    lr_r2_val = r2_score(y_test, lr_preds)
    
    print("\n========================================")
    print("Model Evaluation:")
    print(f"Random Forest - RMSE: {rf_rmse_val:.4f}, R2: {rf_r2_val:.4f}")
    print(f"Linear Regression - RMSE: {lr_rmse_val:.4f}, R2: {lr_r2_val:.4f}")
    print("========================================\n")
    
    # 7. Actual vs Predicted Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, rf_preds, alpha=0.5, color='teal')
    plt.plot([0, 100], [0, 100], '--', color='red', linewidth=2)
    plt.xlabel('Actual Risk Score')
    plt.ylabel('Predicted Risk Score')
    plt.title('Random Forest: Actual vs Predicted Risk')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.savefig('rf_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Feature Importance Plot
    all_features = rf_pipeline.preprocessor.get_feature_names_out()
    importances = rf_pipeline.regressor.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Importances (Random Forest Regressor)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Top 3 Risk Factors:")
    for idx, row in feature_importance_df.head(3).iterrows():
        print(f"- {row['Feature']}: {row['Importance']:.4f}")
    print("========================================\n")
    
    # Save best model
    joblib.dump(rf_pipeline, 'model.pkl')
    print("Model pipeline saved successfully as 'model.pkl'.")

if __name__ == "__main__":
    create_jupyter_notebook()
    run_model_training()
