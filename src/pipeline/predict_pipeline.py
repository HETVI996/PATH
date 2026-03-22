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
        Returns Top-3 career track predictions with confidence scores.

        Returns
        -------
        list of dict, e.g.:
          [
            {"rank": 1, "track": "Technology & Engineering", "confidence": 68.4},
            {"rank": 2, "track": "Data & Research",          "confidence": 21.3},
            {"rank": 3, "track": "Creative & Design",        "confidence":  6.1},
          ]
        """
        try:
            logging.info("Loading artifacts for prediction")

            preprocessor = load_object("artifacts/preprocessor.pkl")
            model        = load_object("artifacts/model.pkl")
            le           = load_object("artifacts/label_encoder.pkl")

            features = features.drop(columns=["Student_ID"], errors="ignore")

            data_scaled = preprocessor.transform(features)

            # Top-3 probabilities
            proba      = model.predict_proba(data_scaled)[0]   # shape: (n_classes,)
            top3_idx   = proba.argsort()[-3:][::-1]
            top3_labels = le.inverse_transform(top3_idx)
            top3_scores = proba[top3_idx] * 100                # convert to %

            results = [
                {
                    "rank":       rank + 1,
                    "track":      label,
                    "confidence": round(float(score), 1),
                }
                for rank, (label, score) in enumerate(zip(top3_labels, top3_scores))
            ]

            logging.info(f"Top-3 predictions: {results}")
            return results

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Collects user survey responses and converts them to a
    single-row DataFrame matching the training data schema.
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
                 Job_Choice: str,
                 No_Money_Problem: str,
                 Fulfillment: str,
                 Success: str,
                 Regret: str,
                 Ideal_Week: str,
                 Environment: str,
                 Failure: str):

        self.Bookstore        = Bookstore
        self.Curiosity        = Curiosity
        self.Flow             = Flow
        self.Childhood        = Childhood
        self.Pride_Project    = Pride_Project
        self.Hobbies          = Hobbies
        self.Energy           = Energy
        self.Friend_Help      = Friend_Help
        self.Math             = Math
        self.Language         = Language
        self.Creativity       = Creativity
        self.Management       = Management
        self.Group_Role       = Group_Role
        self.Work_Rhythm      = Work_Rhythm
        self.Thinking         = Thinking
        self.Structure        = Structure
        self.Decision         = Decision
        self.Job_Choice       = Job_Choice
        self.No_Money_Problem = No_Money_Problem
        self.Fulfillment      = Fulfillment
        self.Success          = Success
        self.Regret           = Regret
        self.Ideal_Week       = Ideal_Week
        self.Environment      = Environment
        self.Failure          = Failure

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                "Bookstore":        [self.Bookstore],
                "Curiosity":        [self.Curiosity],
                "Flow":             [self.Flow],
                "Childhood":        [self.Childhood],
                "Pride_Project":    [self.Pride_Project],
                "Hobbies":          [self.Hobbies],
                "Energy":           [self.Energy],
                "Friend_Help":      [self.Friend_Help],
                "Math":             [self.Math],
                "Language":         [self.Language],
                "Creativity":       [self.Creativity],
                "Management":       [self.Management],
                "Group_Role":       [self.Group_Role],
                "Work_Rhythm":      [self.Work_Rhythm],
                "Thinking":         [self.Thinking],
                "Structure":        [self.Structure],
                "Decision":         [self.Decision],
                "Job_Choice":       [self.Job_Choice],
                "No_Money_Problem": [self.No_Money_Problem],
                "Fulfillment":      [self.Fulfillment],
                "Success":          [self.Success],
                "Regret":           [self.Regret],
                "Ideal_Week":       [self.Ideal_Week],
                "Environment":      [self.Environment],
                "Failure":          [self.Failure],
            }
            df = pd.DataFrame(data_dict)
            logging.info("DataFrame created from user input")
            return df
        except Exception as e:
            raise CustomException(e, sys)


# ── Test block ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        data = CustomData(
            Bookstore        = "Science/Technology",
            Curiosity        = "Technology",
            Flow             = "Coding",
            Childhood        = "Building",
            Pride_Project    = "Built a personal website",
            Hobbies          = "DIY Projects, Robotics",
            Energy           = "Solving complex problems energizes me",
            Friend_Help      = "Technology",
            Math             = 9,
            Language         = 7,
            Creativity       = 8,
            Management       = 6,
            Group_Role       = "Builder",
            Work_Rhythm      = "One big project",
            Thinking         = "Alone",
            Structure        = "Freedom",
            Decision         = "Head",
            Job_Choice       = "Chose higher salary for security",
            No_Money_Problem = "Build sustainable technology",
            Fulfillment      = "Create",
            Success          = "Creating valuable solutions",
            Regret           = "Risk",
            Ideal_Week       = "Deep work with short meetings",
            Environment      = "Startup",
            Failure          = "Not adapting to change",
        )

        df = data.get_data_as_dataframe()
        print("Input Data:\n", df.T.to_string())

        pipeline = PredictPipeline()
        results  = pipeline.predict(df)

        print("\nTop-3 Career Predictions:")
        for r in results:
            print(f"  #{r['rank']}  {r['track']:<35} {r['confidence']}%")

    except Exception as e:
        raise CustomException(e, sys)