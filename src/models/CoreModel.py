import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from .decision_tree import DecisionTree
from .ModelType import ModelType
from .xgboost import XGBoostModel
from .knn import KNNModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class CoreModel:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.model = None
        self.model_type = None


    def load_decision_tree(self, random_state: int = 42, max_depth: int = None, min_samples_split=2, min_samples_leaf=1):
        """Load the Decision Tree model."""
        self.model_type = ModelType.DECISION_TREE

        self.model = DecisionTree(
            random_state=random_state, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf
        )

    def load_xgboost(self, random_state: int = 42, learning_rate: float = 0.1, n_estimators: int = 100, max_depth: int = 3, subsample: float = 1.0, min_child_weight: int = 1, gamma: float = 0, colsample_bytree = 0.7, reg_alpha = 0, reg_lambda = 1):
        """Load the XGBoost model."""
        self.model_type = ModelType.XGBOOST
        self.model = XGBoostModel(random_state=random_state, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample, min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=colsample_bytree, reg_alpha =reg_alpha, reg_lambda = reg_lambda)

    def load_knn(self, n_neighbors: int = 5, weights="uniform" ):
        """Load the KNN model."""
        self.model_type = ModelType.KNN
        self.model = KNNModel(n_neighbors=n_neighbors, weights=weights)

    def get_data(self):
        """Return the DataFrame containing the data."""
        return self.data

    def train(self, X_train, y_train):
        """Train the model based on its type."""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model before training.")

        self.model.train(X_train, y_train)
            

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation and return the mean and standard deviation of the scores."""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model before cross-validation.")

        scores = cross_val_score(self.model.model, X, y, cv=cv)

        return {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std()
        }

    def grid_search(self, X, y, param_grid, cv=5, scoring="accuracy", n_jobs=None):
        """Find the best Decision Tree hyperparameters."""

        grid_search = GridSearchCV(
            estimator=self.model.get_model(),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True
        )

        grid_search.fit(X, y)
        self.model.set_model(grid_search.best_estimator_)


        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_model": grid_search.best_estimator_,
            "results": grid_search.cv_results_
        }
    

    def predict(self, X_test, Y_test):
        y_pred, y_prob = self.model.evaluate(X_test)

        return {
            "accuracy": accuracy_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred),
            "recall": recall_score(Y_test, y_pred),
            "f1_score": f1_score(Y_test, y_pred),
            "roc_auc": roc_auc_score(Y_test, y_prob),
            "confusion_matrix": confusion_matrix(Y_test, y_pred)
        }

    def load_data_from_csv(self, file_path: str):
        """Load data from a CSV file into the model's DataFrame."""
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        """Preprocess the data in the DataFrame."""
        if self.data is not None:
            self.data.drop_duplicates(inplace=True)
        else:
            raise ValueError("Data is not loaded. Please load data before preprocessing.")

    def split_data(self):
        # split the dataset into features and target variable
        X = self.data.drop("Y", axis=1)
        y = self.data["Y"]

        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def scale_feature_data(self, X_train, X_test):
        """Scale the feature data using standardization."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

