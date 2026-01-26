import os 
import sys
from dataclasses import dataclass
from venv import logger
import pandas as pd
import numpy as np

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
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self, numeric_colums):
        '''
        Builds a preprocessor:
        -Numeric - impute median + scale
        -Text - TF - IDF on a combined ' column
        '''
        try:
            logger.info("Building data transformation pipeline..")
            
            #Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Text pipeline
            text_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("tfidf", TfidfVectorizer(
                    stop_words = "english",
                    max_features = 30000,
                    ngram_range = (1, 2)
                ))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_colums),
                    ("text_pipeline", text_pipeline, 'text_column')
                ],
                remainder='drop'
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        '''
        Reads train/test CSVs created from data_ingestion.
        Prepares:
        - X_train_transformed
        - X_test_transformed
        - Saves preprocessor object
        '''
        
        try:
            logger.info("Data Transformation initiated")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")

            # Dataset numeric columns
            numeric_columns = train_df.select_dtypes(exclude=["object"]).columns.tolist()
            object_columns = train_df.select_dtypes(include=["object"]).columns.tolist()
            
            logger.info(f"Numeric columns: {numeric_columns}")
            logger.info(f"Object columns: {object_columns}")
            
            preprocessor_obj = self.get_daata_transformer_object(numeric_columns)
            
            # Fit only on train, transform both
            X_train = preprocessor_obj.fit_transformer_object(train_df)
            X_test = preprocessor_obj.transform(test_df)
            
            # Save Preprocessor 
            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            logger.info("Processor saved at: {self.transformation_config.preprocessor_obj_file_path}")
            logger.info("Data transformation completed successfully.")
            
            
            return X_train, X_test
        
        except Exception as e:
            raise CustomException(e, sys)