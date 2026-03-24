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


# Columns that contain genuine free text (200+ unique values each)
# These are encoded with TF-IDF, not one-hot encoding
TFIDF_COLS = [
    "Pride_Project",
    "Energy",
    "Job_Choice",
    "No_Money_Problem",
    "Success",
    "Ideal_Week",
    "Failure",
]

# Genuinely categorical columns (fixed small value sets)
STRUCTURED_COLS = [
    "Bookstore", "Curiosity", "Flow", "Childhood", "Friend_Help",
    "Group_Role", "Work_Rhythm", "Thinking", "Structure", "Decision",
    "Fulfillment", "Regret", "Environment", "Hobbies",
]

# Numeric Likert-scale columns
NUMERIC_COLS = ["Math", "Language", "Creativity", "Management"]


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Hybrid preprocessing pipeline:

        - Numeric (4 cols):      median impute → standard scale
        - Categorical (14 cols): most-frequent impute → one-hot encode
        - Free text (7 cols):    combined into one string → TF-IDF
                                 (max 2000 features, bigrams, sublinear TF)

        WHY TFIDF NOT OHE FOR FREE TEXT:
          Pride_Project alone has 458 unique values in this dataset.
          One-hot encoding creates 458 sparse columns that the model
          memorises rather than generalises from. TF-IDF extracts
          meaningful word-level signal (e.g. 'hospital', 'law', 'theatre')
          into a dense 2000-feature vector, which generalises far better.
        """
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ])

            # TF-IDF on the concatenated free-text string
            # max_features=2000, bigrams, sublinear_tf reduces dominance of
            # very frequent words like "completed" or "organised"
            tfidf = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline,  NUMERIC_COLS),
                    ("cat", cat_pipeline,  STRUCTURED_COLS),
                    ("txt", tfidf,         "text_combined"),
                ],
                remainder="drop",
            )

            logging.info("Hybrid TF-IDF + OHE preprocessor built")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info(f"Train shape: {train_df.shape}  Test shape: {test_df.shape}")

            drop_cols  = ["Student_ID", "Career_Feel", "Career_Track"]

            X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
            y_train = train_df["Career_Track"]

            X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
            y_test  = test_df["Career_Track"]

            # Combine all free-text columns into one string per row
            # This lets TF-IDF learn cross-column patterns
            X_train["text_combined"] = X_train[TFIDF_COLS].astype(str).agg(" ".join, axis=1)
            X_test["text_combined"]  = X_test[TFIDF_COLS].astype(str).agg(" ".join, axis=1)

            # Drop original free-text cols (replaced by text_combined)
            X_train = X_train.drop(columns=TFIDF_COLS)
            X_test  = X_test.drop(columns=TFIDF_COLS)

            preprocessor = self.get_data_transformer_object()

            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            logging.info(f"Transformed shapes — train: {X_train_t.shape}  test: {X_test_t.shape}")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                object=preprocessor,
            )
            logging.info("Preprocessor saved")

            return X_train_t, y_train, X_test_t, y_test

        except Exception as e:
            raise CustomException(e, sys)