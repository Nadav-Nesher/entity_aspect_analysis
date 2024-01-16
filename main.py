"""
This module serves as the main entry point for the entity and aspect extraction and evaluation project.
It orchestrates the flow of data through different stages of the process.

Imports:
- `load_data` from `input_data`: Function to load data.
- `ModelEntityAspectGeneration` from `model_entity_aspect_generation`: Class responsible for entity and aspect generation.
- `ModelEvaluation` from `model_evaluation`: Class for evaluating the model's performance.

Functions:
- `flow`: Orchestrates the process of loading data, generating entities and aspects, and evaluating the model.
"""


from input_data import load_data
from model_entity_aspect_generation import ModelEntityAspectGeneration
from model_evaluation import ModelEvaluation


def flow():
    """
    Executes the main workflow of the application.

    This function handles the following steps:
    1. Load data from a specified path.
    2. Instantiate the ModelEntityAspectGeneration with the loaded review CSV file.
    3. Instantiate the ModelEvaluation with the loaded data and the generation results.

    Returns:
        An instance of ModelEvaluation containing the review analysis with the
        generation results alongside the evaluation results.
    """
    df = load_data(path='sample_review.csv')
    model_gen_obj = ModelEntityAspectGeneration(df)
    model_eval_obj = ModelEvaluation(df, model_gen_obj)

    return model_eval_obj



if __name__ == '__main__':
    """
    Main execution point of the script. Calls the flow function.
    """
    review_analysis = flow()







