from utils import jaccard_distance

from gensim.models.ldamulticore import LdaMulticore, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.word2vec import LineSentence
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus

from tqdm import tqdm
from pathlib import Path
import numpy as np


def average_jaccard_distance(lda_model:LdaMulticore|LdaModel, topn:int=20) -> float:
    """calculates the average jaccard similarity betwen every topic in an LdaModel

    Args:
        lda_model (LdaMulticore | LdaModel): fit lda model to extract topics
        topn (int, optional): the number of top terms to use per topic in comparison. Defaults to 20.

    Returns:
        float: the average jaccard similarity
    """
    num_topics = lda_model.num_topics
    distances = []
    for topic_num_a in range(num_topics - 1):
        topic_terms_a = [topic[0] for topic in lda_model.show_topic(topic_num_a, topn=topn)]
        for topic_num_b in range(topic_num_a + 1, num_topics):
            topic_terms_b = [topic[0] for topic in lda_model.show_topic(topic_num_b, topn=topn)]
            dist = jaccard_distance(topic_terms_a, topic_terms_b)
            distances.append(dist)
    return np.mean(distances)



def train_lda_model_get_top_topics(train_docs_bow:MmCorpus, text:LineSentence, data_dict:Dictionary, 
                                   num_topics:int, passes:int, iterations:int, eval_every:int|None) -> tuple:
    """trains an lda model and extracts the four supported coherence measures from CoherenceModel

    Args:
        train_docs_bow (MmCorpus): documents in MmCorpus format
        text (LineSentence): text generator in LineSentence format
        data_dict (Dictionary): data dictionary
        num_topics (int): number of topics
        passes (int): number of passes over corpus
        iterations (int): iterations over the doc chunks
        eval_every (int | None): estimate log perplexity after this many steps.

    Returns:
        tuple[LdaMulticore, tuple[float]]: the lda model and the coherence scores for the given number of topics
    """


    lda_model = LdaMulticore(corpus=train_docs_bow,
                            id2word=data_dict,
                            num_topics=num_topics,
                            passes=passes,
                            iterations=iterations,
                            eval_every=eval_every)


    u_mass_coherence_score = CoherenceModel(lda_model, corpus=train_docs_bow, coherence='u_mass').get_coherence()
    c_v_coherence_measure = CoherenceModel(lda_model, texts=text, coherence='c_v').get_coherence()
    c_uci_coherence_measure = CoherenceModel(lda_model, texts=text, coherence='c_uci').get_coherence()
    c_npmi_coherence_measure = CoherenceModel(lda_model, texts=text, coherence='c_npmi').get_coherence()
    
    scores = (
        u_mass_coherence_score,
        c_v_coherence_measure,
        c_uci_coherence_measure,
        c_npmi_coherence_measure
    )

    return lda_model, scores


def tune_topic_num(train_text_path:str|Path, train_docs_bow_path:str|Path, test_docs_bow_path:str|Path, data_dict_path:str|Path, 
                   topic_range_min:int, topic_range_max:int, topn_for_jaccard:int, passes:int, iterations:int, eval_every:int|None=None) -> dict:
    """iterates over a range for the number of topics to test. fit an lda model and extract the coherence scores for each fit.

    Args:
        train_text_path (str | Path): Path to train text file
        train_docs_bow_path (str | Path): Path to train corpus in MmCorpus format
        test_docs_bow_path (str | Path): Path to test corpus in MmCorpus format only used for calculating perplexity
        data_dict_path (str | Path): Path to the built data dictionary
        topic_range_min (int): minimum number of topic in search range inclusive
        topic_range_max (int): maximum number of topics in search range inclusive
        topn_for_jaccard (int): the nubmer of words to include per topic for calculating the average jaccard similarity
        passes (int): number of passes over corpus
        iterations (int): iterations over the doc chunks
        eval_every (int | None, optional): estimate log perplexity after this many steps. Defaults to None.

    Returns:
        dict: coherence scores dictionary per topic range
    """

    train_line_sentence = LineSentence(train_text_path)
    train_docs_bow = MmCorpus(str(train_docs_bow_path))
    test_docs_bow = MmCorpus(str(test_docs_bow_path))
    trigram_dict = Dictionary.load(str(data_dict_path), mmap='r')

    u_mass_list = []
    c_v_list = []
    c_uci_list = []
    c_nmpi_list = []
    perplexity_list = []
    jaccard_list = []
    topic_range = list(range(topic_range_min, topic_range_max + 1))
    
    for num_topics in tqdm(topic_range):

        lda_model, scores = train_lda_model_get_top_topics(train_docs_bow=train_docs_bow,
                                                           text=train_line_sentence, 
                                                           data_dict=trigram_dict, 
                                                           num_topics=num_topics,
                                                           passes=passes,
                                                           iterations=iterations,
                                                           eval_every=eval_every)
        # calculate log perplexity from the test corpus
        perplexity = lda_model.log_perplexity(test_docs_bow)
        # calculate the average jaccard distance
        jaccard_dist = average_jaccard_distance(lda_model, topn_for_jaccard)

        u_mass_list.append(scores[0])
        c_v_list.append(scores[1])
        c_uci_list.append(scores[2])
        c_nmpi_list.append(scores[3])
        perplexity_list.append(perplexity)
        jaccard_list.append(jaccard_dist)


    results_dict = {
        'topics': topic_range,
        'umass_coherence': u_mass_list,
        'c_v_coherence': c_v_list,
        'c_uci_coherence': c_uci_list,
        'c_nmpi_coherence': c_nmpi_list,
        'log_perplexity': perplexity_list,
        'jaccard_distance': jaccard_list
    }
    return results_dict

if __name__ == '__main__':
    from datetime import datetime
    from utils import load_config, save_as_json, random_split_datafile
    from train_lda_model_prep_vis import build_mm_corpus, construct_data_dict
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to config.yaml', type=str)
    args = parser.parse_args()

    # load config
    config = load_config(args.config_path)

    # create files paths
    base_data_folder = Path(config['paths']['base_data_folder'])

    trigram_reviews_path = base_data_folder / 'trigram_reviews.txt'

    base_training_folder = base_data_folder / 'training'
    base_training_folder.mkdir(parents=True, exist_ok=True)

    lda_train_data_path = base_training_folder / 'lda_train.txt'
    lda_test_data_path = base_training_folder / 'lda_test.txt'

    lda_dictionary_path = base_training_folder / 'lda_dict_path'
    train_mmcorp_path = base_training_folder / 'train_mm_corp'
    test_mmcorp_path = base_training_folder / 'test_mm_corp'

    test_results_path = base_training_folder / 'test_results.json'

    start = datetime.now()
    print('Splitting file...')
    # sample from text corpus and split into train, test files
    random_split_datafile(input_file=trigram_reviews_path, 
                          train_file=lda_train_data_path, 
                          validation_file=lda_test_data_path, 
                          sample_size=config['lda_tune']['sample_size'], 
                          test_ratio=config['lda_tune']['test_ratio'])
    
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nConstructing data dict...')
    # construct data dictionary from train data file
    construct_data_dict(prepped_data=lda_train_data_path, 
                        dict_save_path=lda_dictionary_path, 
                        filter_no_below=config['lda']['filter']['no_below'], 
                        filter_no_above=config['lda']['filter']['no_above'])
    
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nBuilding train Mmcorpus...')
    # build train corpus
    build_mm_corpus(trigram_docs_bow_path=train_mmcorp_path, 
                    trigram_revs_path=lda_train_data_path, 
                    data_dict_path=lda_dictionary_path)
    
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nBuilding test Mmcorpus...')
    # build test corpus
    build_mm_corpus(trigram_docs_bow_path=test_mmcorp_path, 
                    trigram_revs_path=lda_test_data_path, 
                    data_dict_path=lda_dictionary_path)
    
    print('Duration: ', str(datetime.now() - start))

    print('\nGetting optimal number of topics...')
    # search over topic range and get results dictionary
    results_dict = tune_topic_num(train_text_path=lda_train_data_path,
                                  train_docs_bow_path=train_mmcorp_path, 
                                  test_docs_bow_path=test_mmcorp_path, 
                                  data_dict_path=lda_dictionary_path, 
                                  topic_range_min=config['lda_tune']['min_topic'], 
                                  topic_range_max=config['lda_tune']['max_topic'],
                                  topn_for_jaccard=config['lda_tune']['topn_for_jaccard'],
                                  passes=config['lda']['passes'],
                                  iterations=config['lda']['iterations'])

    print('\nSaving results...')
    # save results file
    save_as_json(dict_to_save=results_dict, 
                 save_path=test_results_path)