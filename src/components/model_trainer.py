import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")
    class_names_file_path: str   = os.path.join("artifacts", "class_names.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training — 13-track model")

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc  = le.transform(y_test)
            logging.info(f"Classes ({len(le.classes_)}): {le.classes_.tolist()}")

            models = {
                # saga solver handles sparse TF-IDF matrices well
                "Logistic Regression": LogisticRegression(
                    max_iter=3000, C=1.0, class_weight="balanced",
                    solver="saga", n_jobs=-1
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, max_depth=12, min_samples_leaf=3,
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
            }

            best_model = None; best_score = -1; best_model_name = None

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train_enc)

                y_pred    = model.predict(X_test)
                test_acc  = accuracy_score(y_test_enc, y_pred)
                train_acc = accuracy_score(y_train_enc, model.predict(X_train))

                top3_acc = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    top3_acc = sum(
                        1 for i, true in enumerate(y_test_enc)
                        if true in proba[i].argsort()[-3:][::-1]
                    ) / len(y_test_enc)

                print(f"\n{'='*52}")
                print(f"{model_name}")
                print(f"  Train accuracy : {train_acc:.4f}")
                print(f"  Test  accuracy : {test_acc:.4f}")
                if top3_acc: print(f"  Top-3 accuracy : {top3_acc:.4f}")
                if train_acc - test_acc > 0.2:
                    print(f"  [WARNING] Overfit gap: {train_acc - test_acc:.3f}")
                print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

                if test_acc > best_score:
                    best_score = test_acc; best_model = model; best_model_name = model_name

            print(f"\nBest model: {best_model_name}  accuracy={best_score:.4f}")

            if best_model is None:
                raise Exception("No model trained.")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            save_object(self.model_trainer_config.label_encoder_file_path, le)
            save_object(self.model_trainer_config.class_names_file_path,   le.classes_.tolist())

            logging.info("Model, encoder and class names saved.")
            return best_score

        except Exception as e:
            raise CustomException(e, sys)