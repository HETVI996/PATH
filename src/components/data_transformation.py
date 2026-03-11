import os
import sys
from dataclasses import dataclass

import pandas as pd

from sklearn.preprocessing import StandardScaler
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


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numeric_columns):
        """
        Builds preprocessing pipeline:
        - Numeric: median imputation + scaling
        - Text: TF-IDF vectorization
        """

        try:
            logging.info("Building data transformation pipeline")

            # Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Text pipeline
            text_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("tfidf", TfidfVectorizer(
                        stop_words="english",
                        max_features=5000,
                        ngram_range=(1, 2)
                    ))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("text_pipeline", text_pipeline, "text_column")
                ],
                remainder="drop"
            )

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

            target_column = "Career_Fe"

            # Separate target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Identify numeric columns
            numeric_columns = X_train.select_dtypes(exclude=["object"]).columns.tolist()

            logging.info(f"Numeric columns: {numeric_columns}")

            # Combine text columns
            text_columns = ["Pride_Proj", "Hobbies", "Energy"]

            X_train["text_column"] = X_train[text_columns].astype(str).agg(" ".join, axis=1)
            X_test["text_column"] = X_test[text_columns].astype(str).agg(" ".join, axis=1)

            # Create preprocessor
            preprocessor_obj = self.get_data_transformer_object(numeric_columns)

            # Fit on train
            X_train_transformed = preprocessor_obj.fit_transform(X_train)

            # Transform test
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info(
                f"Preprocessor saved at {self.transformation_config.preprocessor_obj_file_path}"
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)