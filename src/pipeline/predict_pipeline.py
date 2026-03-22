import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Takes a user input DataFrame, transforms it, and returns
        predicted career type(s) as human-readable class names.
        """
        try:
            logging.info("Loading preprocessor, model, and label encoder for prediction")

            preprocessor = load_object("artifacts/preprocessor.pkl")
            model = load_object("artifacts/model.pkl")
            label_encoder = load_object("artifacts/label_encoder.pkl")

            # Drop Student_ID if accidentally passed in
            features = features.drop(columns=["Student_ID"], errors="ignore")

            # Transform input using the fitted preprocessor
            data_scaled = preprocessor.transform(features)

            logging.info("Data transformed, making predictions")

            # Predict integer class index
            pred_indices = model.predict(data_scaled)

            # Decode back to string class names
            pred_labels = label_encoder.inverse_transform(pred_indices)

            logging.info(f"Prediction: {pred_labels}")
            return pred_labels

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Collects user survey responses and converts them to a
    single-row DataFrame matching the training data schema.

    FIX for Bug 6: corrected column name mismatches:
      - Job_choice   -> Job_Choice   (capital C)
      - Fullfillment -> Fulfillment  (removed extra 'l')
      - Ideal_week   -> Ideal_Week   (capital W)
    """

    def __init__(self,
                 Bookstore: str,
                 Curiosity: str,
                 Flow: str,
                 Childhood: str,
                 Pride_Project: str,
                 Hobbies: str,
                 Energy: str,
                 Friend_Help: str,
                 Math: int,
                 Language: int,
                 Creativity: int,
                 Management: int,
                 Group_Role: str,
                 Work_Rhythm: str,
                 Thinking: str,
                 Structure: str,
                 Decision: str,
                 Job_Choice: str,          # was: Job_choice
                 No_Money_Problem: str,
                 Fulfillment: str,         # was: Fullfillment (typo)
                 Success: str,
                 Regret: str,
                 Ideal_Week: str,          # was: Ideal_week
                 Environment: str,
                 Failure: str):

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
        self.Job_Choice = Job_Choice
        self.No_Money_Problem = No_Money_Problem
        self.Fulfillment = Fulfillment
        self.Success = Success
        self.Regret = Regret
        self.Ideal_Week = Ideal_Week
        self.Environment = Environment
        self.Failure = Failure

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Converts survey input to a pandas DataFrame matching training schema."""
        try:
            data_dict = {
                "Bookstore": [self.Bookstore],
                "Curiosity": [self.Curiosity],
                "Flow": [self.Flow],
                "Childhood": [self.Childhood],
                "Pride_Project": [self.Pride_Project],
                "Hobbies": [self.Hobbies],
                "Energy": [self.Energy],
                "Friend_Help": [self.Friend_Help],
                "Math": [self.Math],
                "Language": [self.Language],
                "Creativity": [self.Creativity],
                "Management": [self.Management],
                "Group_Role": [self.Group_Role],
                "Work_Rhythm": [self.Work_Rhythm],
                "Thinking": [self.Thinking],
                "Structure": [self.Structure],
                "Decision": [self.Decision],
                "Job_Choice": [self.Job_Choice],
                "No_Money_Problem": [self.No_Money_Problem],
                "Fulfillment": [self.Fulfillment],
                "Success": [self.Success],
                "Regret": [self.Regret],
                "Ideal_Week": [self.Ideal_Week],
                "Environment": [self.Environment],
                "Failure": [self.Failure],
            }

            df = pd.DataFrame(data_dict)
            logging.info("DataFrame created successfully from user input")
            return df

        except Exception as e:
            logging.error("Error while creating DataFrame from user input")
            raise CustomException(e, sys)


# ── Test block ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        data = CustomData(
            Bookstore="Science/Technology",
            Curiosity="Technology",
            Flow="Coding",
            Childhood="Building",
            Pride_Project="Built a personal website",          # must match one of the 10 training values
            Hobbies="DIY Projects, Robotics",                  # must match one of the 10 training values
            Energy="Solving complex problems energizes me",    # must match one of the 10 training values
            Friend_Help="Technology",
            Math=9,
            Language=7,
            Creativity=8,
            Management=6,
            Group_Role="Builder",
            Work_Rhythm="One big project",
            Thinking="Alone",
            Structure="Freedom",
            Decision="Head",
            Job_Choice="Chose higher salary for security",     # must match training value exactly
            No_Money_Problem="Build sustainable technology",
            Fulfillment="Create",
            Success="Creating valuable solutions",
            Regret="Risk",
            Ideal_Week="Deep work with short meetings",
            Environment="Startup",
            Failure="Not adapting to change"
        )

        df = data.get_data_as_dataframe()
        print("Input Data:\n", df.T)

        pipeline = PredictPipeline()
        result = pipeline.predict(df)
        print("\nPredicted Career Feel:", result)

    except Exception as e:
        raise CustomException(e, sys)