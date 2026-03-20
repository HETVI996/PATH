import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation   
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass
        
    def run_pipeline(self):
        
        """
        Full ML Pipeline 
        1> Data Ingestion
        2> Data Transformation  
        3> Model Training
        
        """
        
        try:
            logging.info("Pipeline started")
            
            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            
            logging.info("Data Ingestion completed")
            
            # Step 2: Data Transformation
            transformation = DataTransformation()
            X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(train_path, test_path)
            
            logging.info("Data Transformation completed")
            
            # Step 3: Model Training
            trainer = ModelTrainer()
            trainer.initiate_model_training(X_train, y_train, X_test, y_test)
            
            logging.info("Model Training completed")
        
        
        except Exception as e:
            raise CustomException(e, sys)    
        
        
# Entry Point 
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()