from sklearn.neighbors import KNeighborsClassifier 
from models.BaseModel import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors: int = 5, weights="uniform"):
        """Initialize the KNN model with a specified number of neighbors."""
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))