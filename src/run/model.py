import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .decision_tree import DecisionTree

class Model:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def get_data(self):
        """Return the DataFrame containing the data."""
        return self.data

    def train(self, X_train, y_train, model_type: str = "decision_tree"):
        if model_type == "decision_tree":
            # Training logic for decision tree
            self.model = DecisionTree()
            self.model.train(X_train, y_train)

        elif model_type == "xgboost":
            # Training logic for XGBoost
            pass
        elif model_type == "knn":
            # Training logic for KNN
            pass

    def predict(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

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

