import os 
import sys 
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging      
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        
        try:
            logging.info("Strarting model training")
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth = 5
                ),
                "Gradient Boosting": GradientBoostingClassifier()
            }
            
            model_report = {}
            
            best_model = None
            best_score = 0
            
            for model_name, model in models.items():
                
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)  
                model_report[model_name] = score
                logging.info(f"{model_name} Accuracy: {score}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                
                logging.info(f"Best model score: {best_score}")
                
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                
                logging.info("Model training completed and model saved")
                
                return best_score
                            
        except Exception as e:
            raise CustomException(e, sys)