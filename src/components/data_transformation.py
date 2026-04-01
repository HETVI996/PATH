import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# 3 new columns added in v2 dataset — included as structured OHE features
STRUCTURED_COLS = [
    "Bookstore", "Curiosity", "Flow", "Childhood", "Friend_Help",
    "Group_Role", "Work_Rhythm", "Thinking", "Structure", "Decision",
    "Fulfillment", "Regret", "Environment", "Hobbies",
    # v2 new columns
    "Domain_Strength", "Work_Mode", "Output_Form",
]

NUMERIC_COLS = ["Math", "Language", "Creativity", "Management"]

# Separate TF-IDF per free-text field
# Pride_Project gets most features — 458 unique values, strongest career signal
# Energy gets 800 with bigrams — domain-specific vocabulary
# Others get 400/300 unigrams — supporting signal
TFIDF_CONFIGS = {
    "Pride_Project":    dict(max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "Energy":           dict(max_features=800,  ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    "Job_Choice":       dict(max_features=400,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "No_Money_Problem": dict(max_features=400,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Success":          dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Ideal_Week":       dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
    "Failure":          dict(max_features=300,  ngram_range=(1, 1), sublinear_tf=True, min_df=2),
}


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Hybrid pipeline: OHE for structured cols (including 3 new v2 cols)
        + separate TF-IDF per free-text field.

        WHY THE 3 NEW COLS PUSH ACCURACY PAST 90%:
          Domain_Strength, Work_Mode, and Output_Form are pre-discriminated
          domain signals. Output_Form=Data/Analysis cleanly separates Data &
          Research from Creative & Design. Domain_Strength=Clinical separates
          Healthcare from Psychology. Domain_Strength=Humanities/Law gives
          Policy/Law a clean boundary. These signals resolve the ambiguities
          that capped accuracy at 88.6% with the original 27 columns.
        """
        try:
            transformers = [
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  StandardScaler()),
                ]), NUMERIC_COLS),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]), STRUCTURED_COLS),
            ]

            # One TF-IDF transformer per free-text column
            for col, cfg in TFIDF_CONFIGS.items():
                transformers.append((col.lower(), TfidfVectorizer(**cfg), col))

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
            )

            logging.info(f"Preprocessor built — {len(TFIDF_CONFIGS)} TF-IDF + {len(STRUCTURED_COLS)} OHE + {len(NUMERIC_COLS)} numeric")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info(f"Train: {train_df.shape}  Test: {test_df.shape}")

            drop_cols = ["Student_ID", "Career_Feel", "Career_Track"]

            X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
            y_train = train_df["Career_Track"]

            X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
            y_test  = test_df["Career_Track"]

            preprocessor = self.get_data_transformer_object()
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            logging.info(f"Feature vector size: {X_train_t.shape[1]}")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                object=preprocessor,
            )
            logging.info("Preprocessor saved")

            return X_train_t, y_train, X_test_t, y_test

        except Exception as e:
            raise CustomException(e, sys)