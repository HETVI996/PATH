import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# These columns are passed directly to their individual TF-IDF transformers.
# Do NOT combine them — the preprocessor now has one TF-IDF per field.
TFIDF_COLS = [
    "Pride_Project", "Energy", "Job_Choice",
    "No_Money_Problem", "Success", "Ideal_Week", "Failure",
]


class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        """
        Returns Top-3 career track predictions with confidence scores.
        Each free-text field is passed directly to its own TF-IDF transformer
        inside the saved preprocessor — no pre-processing needed here.
        """
        try:
            preprocessor = load_object("artifacts/preprocessor.pkl")
            model        = load_object("artifacts/model.pkl")
            le           = load_object("artifacts/label_encoder.pkl")

            features = features.drop(columns=["Student_ID"], errors="ignore")

            # Pass features directly — preprocessor handles everything
            data_scaled = preprocessor.transform(features)
            proba       = model.predict_proba(data_scaled)[0]
            top3_idx    = proba.argsort()[-3:][::-1]
            top3_labels = le.inverse_transform(top3_idx)
            top3_scores = proba[top3_idx] * 100

            results = [
                {"rank": rank + 1, "track": label, "confidence": round(float(score), 1)}
                for rank, (label, score) in enumerate(zip(top3_labels, top3_scores))
            ]
            logging.info(f"Top-3: {results}")
            return results

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """All 25 survey fields. Free-text fields accept genuine open text."""

    def __init__(self,
                 Bookstore, Curiosity, Flow, Childhood,
                 Pride_Project, Hobbies, Energy, Friend_Help,
                 Math, Language, Creativity, Management,
                 Group_Role, Work_Rhythm, Thinking, Structure, Decision,
                 Job_Choice, No_Money_Problem, Fulfillment,
                 Success, Career_Feel, Regret, Ideal_Week,
                 Environment, Failure):
        self.Bookstore=Bookstore; self.Curiosity=Curiosity; self.Flow=Flow
        self.Childhood=Childhood; self.Pride_Project=Pride_Project
        self.Hobbies=Hobbies; self.Energy=Energy; self.Friend_Help=Friend_Help
        self.Math=Math; self.Language=Language; self.Creativity=Creativity
        self.Management=Management; self.Group_Role=Group_Role
        self.Work_Rhythm=Work_Rhythm; self.Thinking=Thinking
        self.Structure=Structure; self.Decision=Decision
        self.Job_Choice=Job_Choice; self.No_Money_Problem=No_Money_Problem
        self.Fulfillment=Fulfillment; self.Success=Success
        self.Career_Feel=Career_Feel; self.Regret=Regret
        self.Ideal_Week=Ideal_Week; self.Environment=Environment
        self.Failure=Failure

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
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
                "Career_Feel":      [self.Career_Feel],
                "Regret":           [self.Regret],
                "Ideal_Week":       [self.Ideal_Week],
                "Environment":      [self.Environment],
                "Failure":          [self.Failure],
            })
        except Exception as e:
            raise CustomException(e, sys)