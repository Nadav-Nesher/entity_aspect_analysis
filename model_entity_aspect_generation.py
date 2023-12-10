"""
This module defines the ModelEntityAspectGeneration class, which is responsible for generating named entities
and aspects from customer reviews using the OpenAI API.

Classes:
- ModelEntityAspectGeneration: Handles the process of named entity and aspect generation from customer reviews.

Imports:
- pandas: A data manipulation and analysis library.
- ast: A library to process trees of the Python abstract syntax grammar.
- typing: Provides type hinting capabilities.
- helper_funcs: Contains functions to interact with the OpenAI API.
"""

import pandas as pd
import ast
from typing import Dict, List, Tuple
from helper_funcs import get_completion_from_messages, ResponseFormat

class ModelEntityAspectGeneration:
    """
    A class that handles the generation of named entities and aspects from customer reviews.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the reviews and their true entities and aspects.
        review (str): The review text to be analyzed.
        messages (List[Dict[str, str]]): A list of chat messages formatted for interaction with the OpenAI API.
        review_analysis (dict): The analysis result of the review, containing generated entities and aspects.
        true_entity (List[str]): The true named entities extracted from the DataFrame.
        true_aspect (List[str]): The true aspects extracted from the DataFrame.
        predicted_entity (List[str]): The predicted named entities by the model.
        predicted_aspect (List[str]): The predicted aspects by the model.
        generation (Tuple[str, str]): A tuple containing the predicted entity and aspect.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the ModelEntityAspectGeneration class with a DataFrame containing customer reviews.

        Parameters:
            df (pd.DataFrame): A DataFrame containing customer reviews along with their true entities and aspects.
        """
        self.df: pd.DataFrame = df
        self.review: str = ""
        self.messages: List[Dict[str, str]] = []
        self.review_analysis: dict = dict()
        self.true_entity: List[str] = []
        self.true_aspect: List[str] = []
        self.predicted_entity: List[str] = []
        self.predicted_aspect: List[str] = []
        self.generation: Tuple[str, str] = tuple()

        self.order_chat_messages()
        self.extract_true_val()
        self.extract_tuple()

    def order_chat_messages(self):
        """
        Organizes chat messages in a format suitable for interaction with the OpenAI API for entity and aspect generation.
        """

        system_prompt = """
        You are an NLP customer review analyzer who knows how to find what named entity the review \
        is addressing and the aspect/opinion about this named entity. 
        """

        initial_user_prompt = """
        Your task is to extract both the entity and aspect/opinion that are literally mentioned in the review. /
    
        Pay attention to the following (important):
        - The entity and aspect/opinion MUST be explicitly mentioned in the review. Do NOT infer by yourself.
        - The entity can be more than one word long (e.g., "Chinese restaurant")
        - The aspect/opinion can be more than one word long (e.g., "delicious and fantastic", "amazingly good")
        - There could be more than one entity-aspect pair in the same review (e.g., "The pizza was terrific but the music was bad" -> {pizza: terrific, music: bad})
        - If you can't find the entity or aspect/opinion, return "None".
        - The review can be a full sentence but could also simply be a phrase or utterance.
        - Return the response in the following JSON format: {incremental number: {"named_entity": str, "aspect": List[str]}
    
        Entity - what it is that the customer is referring to in his/her review (e.g., "Chinese restaurant", "service", "meal", "waitress", "food").
        Aspect (opinion) - how the entity is described by the customer (e.g., "great", "amazing", "took too much time to prepare", "patient", "superb").
        """

        assistant_res_1 = """
        I acknowledge that I am a customer review analyzer who knows how to find search review for and find \
        both the named entity mentioned in the review and the aspect/opinion addressed to the named entity. \
        I understand your request and will accordingly look for and find both the named entity and the aspect/opinion
        """

        user_clarification_prompt = """
        Below are a few examples for you to learn from (few-shot learning):
        "The food is decent" --> food (entity), decent (aspect)
        "The Steak Tartare was splendid" --> Steak Tartare (entity), splendid (aspect)
        "The service is top-notch" --> service (entity), top-notch (aspect)
        "I had the duck breast special on my last visit and it was incredible." --> duck breast special (entity), incredible (aspect)
        "The hostess was extremely rude and offensive" --> hostess (entity), extremely rude and offensive (aspect)
        "Chow fun was dry ; pork shu mai was more than usually greasy and had to share a table with loud and rude family." --> Chow fun (entity), dry (aspect); pork shu mai (entity), more than usually greasy (aspect); table (entity), had to share with loud and rude family (aspect)  
        "The waiter took his time with the food" --> waiter (entity), took his time with the food (aspect)
        "Ambience is delightful, service impeccable." --> ambience (entity), delightful (aspect); service (entity) impeccable (aspect)
        "I won't come back again" --> None (entity), None (aspect)
        "We, there were four of us, arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude." --> place (entity), empty (aspect); staff (entity) acted like we were imposing on them (aspect); staff (entity) very rude (aspect)
        The food is very average . . .the Thai fusion stuff is a bit too sweet , every thing they serve is too sweet here. --> food (entity), very average (aspect); Thai fusion stuff (entity), a bit too sweet (aspect); everything they serve (entity), too sweet (aspect);
        The only thing I moderately enjoyed was their Grilled Chicken special with Edamame Puree. --> Grilled Chicken special with Edamame Puree (entity), moderately enjoyed (aspect);
        """

        assistant_res_2 = """
        Please provide the review you want me to analyze.
        """

        self.review = f"""
        {self.df.loc[0,'review']}
        """

        self.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': initial_user_prompt},
            {'role': 'assistant', 'content': assistant_res_1},
            {'role': 'user', 'content': user_clarification_prompt},
            {'role': 'assistant', 'content': assistant_res_2},
            {'role': 'user', 'content': self.review}
        ]

        review_analysis = get_completion_from_messages(messages=self.messages, response_format=ResponseFormat.JSON_OBJECT)
        self.review_analysis = ast.literal_eval(review_analysis)

    def extract_true_val(self):
        """
        Extracts the true entity and aspect values from the DataFrame for comparison with generated values.
        """
        self.true_entity = [self.df['true_entity'][0]]
        self.true_aspect = [self.df['true_aspect'][0]]

    def extract_tuple(self):
        """
        Extracts the predicted entity and aspect from the analysis and updates the corresponding class attributes.

        This method updates:
            - self.generation: A tuple containing the predicted entity and aspect.
            - self.predicted_entity: A list containing the predicted named entity.
            - self.predicted_aspect: A list containing the predicted aspect.
        """
        predicted_entity = self.review_analysis["1"]["named_entity"]
        predicted_aspect = self.review_analysis["1"]["aspect"][0]
        self.generation = (predicted_entity, predicted_aspect)
        self.predicted_entity = [predicted_entity]
        self.predicted_aspect = [predicted_aspect]
