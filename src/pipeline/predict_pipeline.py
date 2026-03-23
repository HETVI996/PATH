import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.text_classifier import classify_row


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Takes a user input DataFrame (with free-text answers for Q5,Q6,Q7,
        Q15,Q16,Q18,Q19,Q21,Q23), classifies the free-text into training
        labels, then returns Top-3 career track predictions with confidence.

        Returns
        -------
        list of dict:
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

            # Drop ID column if present
            features = features.drop(columns=["Student_ID"], errors="ignore")

            # ── Classify free-text answers ────────────────────────────────────
            # Convert each row to dict, run classifier, rebuild DataFrame
            classified_rows = [classify_row(row) for row in features.to_dict(orient="records")]
            features_classified = pd.DataFrame(classified_rows)

            logging.info(f"Classified free-text columns: {features_classified.to_dict(orient='records')[0]}")

            # ── Transform + predict ───────────────────────────────────────────
            data_scaled = preprocessor.transform(features_classified)
            proba       = model.predict_proba(data_scaled)[0]
            top3_idx    = proba.argsort()[-3:][::-1]
            top3_labels = le.inverse_transform(top3_idx)
            top3_scores = proba[top3_idx] * 100

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
    Collects all 25 survey responses and converts them to a DataFrame.

    Free-text questions (Q5, Q6, Q7, Q15, Q16, Q18, Q19, Q21, Q23)
    now accept genuine free text — the text_classifier maps them to
    training labels automatically inside PredictPipeline.predict().

    Structured questions (Q1–Q4, Q8–Q14, Q17, Q20, Q22) still expect
    one of the fixed option values exactly as shown in the survey.
    """

    def __init__(self,
                 Bookstore: str,
                 Curiosity: str,
                 Flow: str,
                 Childhood: str,
                 Pride_Project: str,       # free text — Q5
                 Hobbies: str,             # free text — Q6
                 Energy: str,              # free text — Q7
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
                 Job_Choice: str,          # free text — Q15
                 No_Money_Problem: str,    # free text — Q16
                 Fulfillment: str,
                 Success: str,             # free text — Q18
                 Regret: str,
                 Ideal_Week: str,          # free text — Q21 (Career_Feel in Q19 is internal)
                 Environment: str,
                 Failure: str):            # free text — Q23

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
        # Test with genuine free-text answers (not pre-bucketed)
        data = CustomData(
            Bookstore        = "Science/Technology",
            Curiosity        = "Technology",
            Flow             = "Coding",
            Childhood        = "Building",
            Pride_Project    = "I built a patient management system for a hospital",
            Hobbies          = "I enjoy going to the gym, playing cricket, and cooking healthy meals",
            Energy           = "I feel most alive when I solve really hard algorithm problems",
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
            Job_Choice       = "I chose the job with more learning opportunities over higher salary",
            No_Money_Problem = "I would solve the mental health crisis among young people",
            Fulfillment      = "Create",
            Success          = "Building products that change how people live",
            Regret           = "Risk",
            Ideal_Week       = "Deep focus coding sessions with short daily standups",
            Environment      = "Startup",
            Failure          = "Starting many projects but never delivering a finished product",
        )

        df = data.get_data_as_dataframe()
        print("Raw input (free text):")
        print(f"  Pride_Project    : {df['Pride_Project'].values[0]}")
        print(f"  Hobbies          : {df['Hobbies'].values[0]}")
        print(f"  Energy           : {df['Energy'].values[0]}")
        print(f"  Job_Choice       : {df['Job_Choice'].values[0]}")
        print(f"  No_Money_Problem : {df['No_Money_Problem'].values[0]}")
        print()

        pipeline = PredictPipeline()
        results  = pipeline.predict(df)

        print("Top-3 Career Predictions:")
        for r in results:
            print(f"  #{r['rank']}  {r['track']:<35} {r['confidence']}%")

    except Exception as e:
        raise CustomException(e, sys)