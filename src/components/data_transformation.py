import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numeric_columns, categorical_columns):
        """
        Builds preprocessing pipeline:
        - Numeric: median imputation + standard scaling
        - Categorical: most-frequent imputation + one-hot encoding

        WHY no TF-IDF:
          Pride_Project, Hobbies, Energy each have only 10 fixed string
          values — they are categorical, not free text. One-hot encoding
          captures their full information cleanly. TF-IDF on fixed-vocabulary
          strings adds noise and misses the categorical structure.
        """
        try:
            logging.info("Building data transformation pipeline")

            # Numeric pipeline: impute then scale to zero-mean unit-variance
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline: impute then one-hot encode
            # handle_unknown='ignore' ensures unseen values at prediction time
            # become an all-zero row instead of raising an error
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # FIX for Bug 1: remainder="drop" was silently discarding all 21
            # categorical columns. Now every column is explicitly assigned to
            # one of the two transformers — nothing is dropped.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop"   # safe now: all columns are covered above
            )

            logging.info("Data transformation pipeline built successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Data Transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            target_column = "Career_Feel"

            # Separate target
            X_train = train_df.drop(columns=[target_column, "Student_ID"], errors="ignore")
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column, "Student_ID"], errors="ignore")
            y_test = test_df[target_column]

            # Split columns by dtype
            # Numeric: Math, Language, Creativity, Management (Likert 4-9)
            numeric_columns = X_train.select_dtypes(exclude=["object"]).columns.tolist()

            # Categorical: all remaining columns, including the previously
            # mishandled semi-structured ones (Pride_Project, Hobbies, Energy,
            # Job_Choice, No_Money_Problem, Success, Ideal_Week, Failure,
            # Career_Feel) — each has exactly 10 fixed string values
            categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()

            logging.info(f"Numeric columns  ({len(numeric_columns)}): {numeric_columns}")
            logging.info(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

            # Build and fit preprocessor
            preprocessor_obj = self.get_data_transformer_object(numeric_columns, categorical_columns)

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            logging.info(f"Transformed train shape: {X_train_transformed.shape}")
            logging.info(f"Transformed test shape:  {X_test_transformed.shape}")

            # Save preprocessor for use in prediction pipeline
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                object=preprocessor_obj
            )

            logging.info(
                f"Preprocessor saved at {self.transformation_config.preprocessor_obj_file_path}"
            )

            return (
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)