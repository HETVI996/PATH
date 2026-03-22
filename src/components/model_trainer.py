import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training")

            # Encode string labels to integers for all sklearn classifiers
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            logging.info(f"Classes: {le.classes_.tolist()}")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    min_samples_leaf=3, random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=8, min_samples_leaf=5, random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=4, random_state=42
                ),
            }

            model_report = {}
            best_model = None
            best_score = -1
            best_model_name = None

            # FIX for Bugs 4 & 5:
            # Previously both save_object() and return were INSIDE the loop,
            # which meant:
            #   - the function returned after the very first model (LR only)
            #   - the model file was overwritten on every iteration
            # Now: train ALL models first, track the best, then save once.

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train_enc)

                # Test accuracy
                y_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test_enc, y_pred)

                # Train accuracy (overfitting check)
                y_train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train_enc, y_train_pred)

                print(f"\n{'='*50}")
                print(f"{model_name}")
                print(f"  Train accuracy : {train_acc:.4f}")
                print(f"  Test  accuracy : {test_acc:.4f}")
                if train_acc - test_acc > 0.2:
                    print(f"  [WARNING] Gap {train_acc - test_acc:.3f} — model may be overfitting")

                print(f"\n{model_name} Classification Report:")
                print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

                logging.info(f"{model_name} — train={train_acc:.4f}  test={test_acc:.4f}")

                model_report[model_name] = test_acc

                if test_acc > best_score:
                    best_score = test_acc
                    best_model = model
                    best_model_name = model_name

            # ---- Save AFTER the loop, only once, only the best model ----
            logging.info(f"Best model: {best_model_name} (test accuracy={best_score:.4f})")
            print(f"\nBest model: {best_model_name}  test accuracy={best_score:.4f}")

            if best_model is None:
                raise Exception("No model was trained — check that X_train is non-empty.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            # Also save the label encoder so prediction pipeline can decode
            # integer predictions back to string class names
            save_object(
                file_path=self.model_trainer_config.label_encoder_file_path,
                object=le
            )

            logging.info("Best model and label encoder saved successfully")
            return best_score

        except Exception as e:
            raise CustomException(e, sys)