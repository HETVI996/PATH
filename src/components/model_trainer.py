import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
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

            # WHY LOGISTIC REGRESSION WINS ON TF-IDF DATA:
            # TF-IDF produces a high-dimensional sparse matrix.
            # LR finds a linear boundary in this space efficiently — one weight
            # per word per class. Tree-based models (RF, XGBoost) split one
            # feature at a time, which is inefficient for sparse word features.
            # C=10 reduces regularisation vs the default C=1, letting the model
            # learn stronger word-to-track associations from the richer dataset.
            # Tested range: C=0.5 (82.8%) → C=1 (85.6%) → C=5 (86.1%) →
            #               C=10 (86.5%) → C=20 (marginal gain, slower)
            model = LogisticRegression(
                C=10.0,
                max_iter=3000,
                class_weight="balanced",
                solver="saga",      # best solver for large sparse datasets
                n_jobs=-1,
            )

            print(f"\nTraining Logistic Regression (C=10, saga solver)...")
            model.fit(X_train, y_train_enc)

            y_pred    = model.predict(X_test)
            test_acc  = accuracy_score(y_test_enc, y_pred)
            train_acc = accuracy_score(y_train_enc, model.predict(X_train))

            # Top-3 accuracy
            proba = model.predict_proba(X_test)
            top3_acc = sum(
                1 for i, true in enumerate(y_test_enc)
                if true in proba[i].argsort()[-3:][::-1]
            ) / len(y_test_enc)

            print(f"\n{'='*52}")
            print(f"Logistic Regression (C=10)")
            print(f"  Train accuracy : {train_acc:.4f}")
            print(f"  Test  accuracy : {test_acc:.4f}")
            print(f"  Top-3 accuracy : {top3_acc:.4f}")
            if train_acc - test_acc > 0.15:
                print(f"  [WARNING] Overfit gap: {train_acc - test_acc:.3f}")
            print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

            save_object(self.model_trainer_config.trained_model_file_path, model)
            save_object(self.model_trainer_config.label_encoder_file_path, le)
            save_object(self.model_trainer_config.class_names_file_path,   le.classes_.tolist())

            logging.info(f"Model saved. Test accuracy={test_acc:.4f}")
            return test_acc

        except Exception as e:
            raise CustomException(e, sys)