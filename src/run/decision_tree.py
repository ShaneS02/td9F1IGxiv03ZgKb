from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix)



class DecisionTree:
    def __init__(self, random_state: int = 42):
        """Initialize the Decision Tree model with a random state."""
        self.random_state = random_state
        self.model = DecisionTreeClassifier(random_state=random_state)


    def get_data(self):
        """Return the DataFrame containing the data."""
        return self.data

    def train(self, X_train, y_train):
        """Train the Decision Tree model."""
        self.model.fit(X_train, y_train)

    def _predict(self, X_test):
        """Make predictions using the trained Decision Tree model."""
        return self.model.predict(X_test)

    def _predict_proba(self, X):
        """Generate class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance on the test set."""

        y_pred = self._predict(X_test)
        y_prob = self._predict_proba(X_test)[:, 1]

        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
