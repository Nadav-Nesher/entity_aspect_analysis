## Project Overview
Entity + opinion (aspect) extraction from customer service reviews alongside an evaluation against true annotated labels.


## Dataset:
- **Description:** Yelp customer service restaurant reviews
- **Name:** `train.csv`
- **Content:** The CSV file contains two columns (review, sentiment/opinion/aspect label)
- **Note:** Only the raw reviews were analysed. The sentiment/opinion label was disregarded as the purpose of the code was to find improved ways of extracting it.
- **Note:** This file was used only for EDA purposes.

## True labels
* The dataset did not undergo human annotation for generating true labels.
* A `sample_review.csv` file (containing a single annotated review) was created for a POC comparison of true-vs-pred labels.
 

## Project Parts:
### Part I - EDA
### Part II - Entity opinion tuple extraction
This module is concerned with extracting the entity being reviewed (e.g., 'food') alongside the reviewer's opinion concerning the entity (e.g., 'somewhat tasty').
### Part III - Evaluation
This module is concerned with the evaluation of the entity-opinion tuple extraction against ground truth annotations.


## EDA: Initial and General Insights
* **Num of review:** 1121
* **Sentiment:** ~60% (POS); ~20% (NEUTRAL); ~20% (NEG) 
Most people show positive emotions regarding their dining experience.
![sentiment_distribution](/images/sentiment_distribution.png)
* **Word length:** Approx. 50% of the reviews are 6-15 tokens long.
Most people don’t take the time to write a “very thorough” review.
![word_length](/images/word_length.png)
* **Extended EDA -** Additional NLP techniques that can be used for gaining insights:
Q&A model, WordCloud, N-gram frequency count (after stopwords removal), fuzzy matching, exact/partial matching (stopwords, morphological prefixes and suffixes), clustering algorithms (lexicosyntactic, semantic), sentiment, polarity, aspect and sentiment combined, NER, distinction into food/beverages, distinction into types of meals (breakfast, lunch, dinner), binary POS-based distinction of tokens (adjectives vs rest), adverbial intensifier analysis, syntactic dependency analysis, distribution of punctuations (types, binary, combination with sentiment).


## Code
* **EDA -** See `EDA.ipynb` file
* **Entity opinion extraction + true-vs-pred eval. -** project (`.py` modules)

# Instructions
Run the `main.py` module

**Notes:**
* Make sure to create an `.env` file for storing your OpenAI api key.
* **Use the following formate:** `OPENAI_API_KEY = "your_openai_api_key"`


# Implementation and alternatives:
## EDA
### Simplistic approaches (quick, provide merely an initial macro view of the data)
* Distribution of word count (both absolute and normalized) to measure the overall length of reviews.
* Word cloud

### A more in-depth understanding
Moving from lexicosyntactic to a semantics-pragmatics interface approach
* Topic modelling with BERTopic

## Entity-Opinion Extraction
* NER models are confined to a set of trained entities that do not reflect the full spectrum of possible entities in all domains (e.g., restaurant reviews). They require a tedious process of annotations, reviewing and training.
* Q&A models are mostly trained on a specific set of questions and datasets and can have more type I and II errors than LLMs.

## Prompt Engineering
Emphasis was placed on the following:
* CoT (Chain of Thought)
* Few-shot learning
* System/user/assistant role distinction
* Reassuring the model

## Evaluation Approach: LLM extraction against Ground truth annotations
BERT Sentence-Transformers for measuring the semantic textual similarity using cosine similarity.

After testing various linguistic variations of true-vs-predicted entities and opinions (aspects), a threshold of 0.85 was set for accepting the model’s prediction into production.

### Evaluation Alternatives

#### Sentence-Transformers alongside rule-based solutions
Combining sophisticated approaches (like Sentence-Transformers) with traditional, rule-based approaches in order to strengthen the validity of the results.

For example, transformer-based solutions combined with syntactic dependency parsing, POS and NER (linguistic features) could assist in making sure that “restaurant” and “the restaurant” are regarded as the same (with/out the determiner) as opposed to “the tasty” and “somewhat tasty” example from above.

#### Traditional classification metrics (recall, precision, F1-score, accuracy)
More traditional classification metrics (recall, precision, F1-score, accuracy) are less relevant in this case.

Many of the confusion matrix components are irrelevant in a binary classification task of only one word/phrase out of a whole review (once for entity recognition and once for aspect). This makes it impossible to calculate some of these metrics.

Additionally, these metrics fail to fully capture minute semantics influenced by the inclusion/removal of adjectival/noun modifiers and adverbial intensifiers. Take for example, “tasty” vs “somewhat tasty”.

Moreover, this raises the question of whether to take into account vs disregard partial matches such as these when using recall, precision, F1-score, and accuracy.

#### Traditional N-gram based metrics (lexicosyntactic similarity)
I do believe, however, that traditional N-gram based metrics could be employed for establishing a baseline:
* BLEU (precision-focused)
* ROUGE (recall-focused)
* METEOR (calculates the harmonic mean of precision and recall alongside penalties for word order and phrase differences)

# Solution viability in production
Although it does require more thorough testing, the cosine similarity approach used to measure semantic textual similarity between BERT Sentence-Transformers embeddings alongside a relatively strict threshold (that should mostly take into account the presence/absence of adjectival/noun modifiers and adverbial intensifiers), could work in a production environment.

There are also additional considerations to a production environment, such as latency, computation time, hosting heavy models on servers, and additional costs.