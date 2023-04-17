# Word2Vec-and-Topic-Analysis-Yelp-Reviews
Explores The yelp dataset with an LDA topic analysis and a Word2Vec model using spaCy and Gensim

## Data:
This analysis is based on the [yelp dataset](https://www.yelp.com/dataset). More specifically the `yelp_academic_dataset_reviews.json` file. These reviews were filtered by category, this can be seen in `extract_business_ids_from_database.ipynb` (see Important Note below) where we create a text file that contains business ids linked to reviews to include in the analysis.

<b>Important Note:</b> The business ids file was created using `extract_business_ids_from_database.ipynb` must be run in the folder for [Yelp_sqlite_database](https://github.com/Alkoopman85/Yelp_sqlite_database). This project can be run without filtering with a small change to `config.yaml` file, more on this below.

## Setup:
Adjust the `config.yaml` file as needed. In the paths section set the base_data folder and set the paths to `yelp_academic_dataset_reviews.json` file and if applicable the path to the `business_idx.txt` file. If no filtering is desired then leave that entry blank.

## Replication:
1.) First we need to prepare the text for modeling. This can be done by running `prep_text.py` from the command line with 
```
python prep_text.py config.yaml
```
The result of which can be seen in `inspect_prepared_text.ipynb`. Parameters can be updated/changed in the 'data_prep' section of the config file.

2.) <b> Optional Step: </b> Next we search for the optimal number of topics by running `search_for_best_num_topics.py` from the command line with
```
python search_for_best_num_topics.py config.yaml
```
And the results of the search can be seen in `num_topics_search_results.ipynb`. Parameters can be updated/changed in the 'lda_tune' and 'lda' sections of the config file.

3.) At this stage we are ready to train the LDA model using the optimal number of topics obtained from step 2. This is done by running `train_lda_model_prep_vis.py` from the command line with
```
python train_lda_model_prep_vis.py config.yaml
```
The results, analysis and visualization can be seen in `lda_yelp_reviews.ipynb`. Parameters can be updated/changed in the 'lda' section of the config file.

4.) The final step is to train the word2vec model. To do this we can run the `train_word2vec.py` file from the command line with
```
python train_word2vec.py config.yaml
```
We can inspect the model results and analysis in `Yelp_2_Vec_rsults.ipynb`. Parameters can be updated/changed in the 'word_2_vec' section of the config file.

Note: The .py files take a significant amount of time to run.

## Attributions:
This project is based on this [notebook](https://github.com/pwharrison/modern-nlp-in-python) which is a great guide to using these nlp models in python.

Other helpful links:<br>
https://spacy.io/<br>
https://radimrehurek.com/gensim/