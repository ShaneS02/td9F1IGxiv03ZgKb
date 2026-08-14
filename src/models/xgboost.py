from models.BaseModel import BaseModel
from xgboost import XGBClassifier

class XGBoostModel(BaseModel):
    def __init__(self, 
                    random_state: int = 42,
                    learning_rate: float = 0.1,
                    n_estimators: int = 100,
                    max_depth: int = 3,
                    subsample: float = 1.0,
                    min_child_weight: int = 1,
                    gamma: float = 0
                ):
        super().__init__(XGBClassifier(random_state=random_state, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators, 
                                       max_depth=max_depth, 
                                       subsample=subsample, 
                                       min_child_weight=min_child_weight, 
                                       gamma=gamma
                                       ))