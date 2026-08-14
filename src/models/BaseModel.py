class BaseModel:
    def __init__(self, model):
        self.model = model

    def get_model(self):
            """Return the underlying Decision Tree model."""
            return self.model
    
    def set_model(self, model):
        """Set the underlying Decision Tree model."""
        self.model = model

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

    def evaluate(self, X_test):
        """Evaluate the model's performance on the test set."""

        y_pred = self._predict(X_test)
        y_prob = self._predict_proba(X_test)[:, 1]

        return y_pred, y_prob