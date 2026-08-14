from models.BaseModel import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseModel):
    def __init__(self, random_state: int = 42, max_depth: int = None, min_samples_split=2, min_samples_leaf=1):
        """Initialize the Decision Tree model with a random state."""
        self.random_state = random_state
        super().__init__(
            DecisionTreeClassifier(
                random_state=random_state, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf
            )
        )