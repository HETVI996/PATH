import sys 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features: pd.DataFrame):
        '''
        Takes user input 
        returns the predicted career field
        '''
        
        try:
           logging.info("Loading preprocessor and model for prediction")
           
           # Load saved objects
           preprocessor_path = "artifacts/preprocessor.pkl"
           model_path = "artifacts/model.pkl"
           
           preprocessor = load_object(preprocessor_path)
           model = load_object(model_path)
           
           # Transform input 
           data_scaled = preprocessor.transform(features)
           
           logging.info("Data transformed successfully, making predictions")
           
           # Predict using the model
           
           preds = model.predict(data_scaled)
           
           logging.info("Prediction completed successfully")
           
           return preds
       
            
        except Exception as e:
            raise CustomException(e, sys)
        
        # Custom input class (user -> DataFrame )
        
class CustomData:
    def __init__(self,
                 Bookstore,
                 Curiosity,
                 Flow,
                 Childhood,
                 Pride_Project,
                 Hobbies,
                 Energy,
                 Friend_Help,
                 Math,
                 Language,
                 Creativity,
                 Management,
                 Group_Role,
                 Work_Rhythm,
                 Thinking,
                 Structure,
                 Decision,
                 Job_choice,
                 No_Money_Problem,
                 Fullfillment,
                 Success,
                 Regret,
                 Ideal_week,
                 Environment,
                 Failure):
        
                 self.Bookstore = Bookstore
                 self.Curiosity = Curiosity
                 self.Flow = Flow
                 self.Childhood = Childhood
                 self.Pride_Project = Pride_Project
                 self.Hobbies = Hobbies
                 self.Energy = Energy
                 self.Friend_Help = Friend_Help
                 self.Math = Math
                 self.Language = Language
                 self.Creativity = Creativity
                 self.Management = Management
                 self.Group_Role = Group_Role
                 self.Work_Rhythm = Work_Rhythm
                 self.Thinking = Thinking
                 self.Structure = Structure
                 self.Decision = Decision
                 self.Job_choice = Job_choice
                 self.No_Money_Problem = No_Money_Problem
                 self.Fullfillment = Fullfillment
                 self.Success = Success
                 self.Regret = Regret
                 self.Ideal_week = Ideal_week
                 self.Environment = Environment
                 self.Failure = Failure
                 
    def get_data_as_dataframe(self):
        '''
        Conerts Input into pandas DataFrame
        '''
        
        try:
            data_dict = {
                'Bookstore': [self.Bookstore],
                'Curiosity': [self.Curiosity],
                'Flow': [self.Flow],
                'Childhood': [self.Childhood],
                'Pride_Project': [self.Pride_Project],
                'Hobbies': [self.Hobbies],
                'Energy': [self.Energy],
                'Friend_Help': [self.Friend_Help],
                'Math': [self.Math],
                'Language': [self.Language],
                'Creativity': [self.Creativity],
                'Management': [self.Management],
                'Group_Role': [self.Group_Role],
                'Work_Rhythm': [self.Work_Rhythm],
                'Thinking': [self.Thinking],
                'Structure': [self.Structure],
                'Decision': [self.Decision],
                'Job_choice': [self.Job_choice],
                'No_Money_Problem': [self.No_Money_Problem],
                'Fullfillment': [self.Fullfillment],
                'Success': [self.Success],
                'Regret': [self.Regret],
                'Ideal_week': [self.Ideal_week],
                'Environment': [self.Environment],
                'Failure': [self.Failure]
            }
            
            df = pd.DataFrame(data_dict)
            logging.info("DataFrame created successfully from user input")
            return df
        except Exception as e:
            logging.error("Error occurred while creating DataFrame from user input")
            raise CustomException(e, sys)
        

# Testing Block

if __name__ == "__main__":
    try:
        # Create a sample input
        data = CustomData(
            Bookstore = "Science/Technology",
            Curiosity = "Technology",
            Flow = "Coding",
            Childhood = "Building",
            Pride_Project = "Built a Project",
            Hobbies = "Coding, reading, gaming, drawing and sports",
            Energy = "Solving Problem",
            Friend_Help = "Technology",
            Math = 9,
            Language = 7,
            Creativity = 8,
            Management = 6,
            Group_Role = "The Builder",
            Work_Rhythm = "Focus on one task",
            Thinking = "Working Alone",
            Structure = "Flexible",
            Decision = "Head",
            Job_choice = "High salary",
            No_Money_Problem = "Building Something Impactful",
            Fullfillment = "Creating things",
            Success = "Impact",
            Regret = "Not taking risks",
            Ideal_week = "Balanced ",
            Environment = "Startup",
            Failure = "Learn and retry"  
        )
        
        # Convert to DataFrame
        df = data.get_data_as_dataframe()
        
        print("Input Data:\n", df)
        
        text_columns = ["Pride_Project", "Hobbies", "Energy"]

        df["text_column"] = df[text_columns].astype(str).agg(" ".join, axis=1)
        
        # Predict
        pipeline = PredictPipeline()
        result = pipeline.predict(df)
        
        print("\nPredicted Career Field:\n", result)
        
    except Exception as e:
        raise CustomException(e, sys)