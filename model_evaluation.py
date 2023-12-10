"""
This module defines the ModelEvaluation class, which evaluates the performance of entity and aspect predictions
against the ground truth annotations using cosine similarity scores.

Classes:
- ModelEvaluation: Handles the evaluation of predicted named entities and aspects against true values.

Imports:
- pandas: A data manipulation and analysis library.
- sentence_transformers: A library for sentence-level embeddings.
"""

import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


class ModelEvaluation:
    """
    A class that handles the evaluation of predicted named entities and aspects using cosine similarity scores.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the reviews and their true entities and aspects.
        model_gen_object: An instance of ModelEntityAspectGeneration containing the generated predictions.
        threshold (float): The threshold for cosine similarity score to accept the predictions.
        entity_cosine_score (float): The cosine similarity score for the entity predictions.
        aspect_cosine_score (float): The cosine similarity score for the aspect predictions.
        keep_entity_prediction (bool): Indicates whether to keep the entity prediction based on the evaluation.
        keep_aspect_prediction (bool): Indicates whether to keep the aspect prediction based on the evaluation.
    """

    def __init__(self, df: pd.DataFrame, model_gen_object, threshold_score: float = 0.85):
        """
        Initializes the ModelEvaluation class with a DataFrame, a model generation object, and an optional threshold score.

        Parameters:
            df (pd.DataFrame): A DataFrame containing customer reviews along with their true entities and aspects.
            model_gen_object: An instance of ModelEntityAspectGeneration.
            threshold_score (float): The threshold for cosine similarity score to consider a prediction as acceptable.
        """
        self.df: pd.DataFrame = df
        self.model_gen_object = model_gen_object
        self.threshold = threshold_score
        self.entity_cosine_score: float = 0
        self.aspect_cosine_score: float = 0
        self.keep_entity_prediction: bool = None
        self.keep_aspect_prediction: bool = None

        self.invoke_extract_cosine_score_logic()
        self.invoke_evaluation_logic()


    def extract_cosine_score(self, true_val: List[str], predicted_val: List[str]) -> float:
        """
        Computes the cosine similarity score between the true and predicted values.

        Parameters:
            true_val (List[str]): A list of true values (entities or aspects).
            predicted_val (List[str]): A list of predicted values (entities or aspects).

        Returns:
            float: The cosine similarity score between the true and predicted values.
        """
        # Compute embedding for both lists
        embeddings_true = model.encode(true_val, convert_to_tensor=True)
        embeddings_pred = model.encode(predicted_val, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores_tensor = util.cos_sim(embeddings_true, embeddings_pred)
        cosine_score = cosine_scores_tensor[0, 0].item()

        return cosine_score

    def invoke_extract_cosine_score_logic(self):
        """
        Invokes the extract_cosine_score method for both entities and aspects and stores the results.
        """
        self.entity_cosine_score = self.extract_cosine_score(self.model_gen_object.true_entity, self.model_gen_object.predicted_entity)
        self.aspect_cosine_score = self.extract_cosine_score(self.model_gen_object.true_aspect, self.model_gen_object.predicted_aspect)

    def evaluate(self, cosine_score: float) -> bool:
        """
        Evaluates if the cosine similarity score meets or exceeds the threshold.

        Parameters:
            cosine_score (float): The cosine similarity score to be evaluated.

        Returns:
            bool: True if the score meets or exceeds the threshold, False otherwise.
        """

        if cosine_score >= self.threshold:
            return True
        else:
            return False

    def invoke_evaluation_logic(self):
        """
        Invokes the evaluate method for both entity and aspect cosine scores and stores the results.
        """
        self.keep_entity_prediction = self.evaluate(cosine_score=self.entity_cosine_score)
        self.keep_aspect_prediction = self.evaluate(cosine_score=self.aspect_cosine_score)