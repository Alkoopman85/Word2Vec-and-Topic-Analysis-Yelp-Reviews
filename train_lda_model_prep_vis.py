from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import LineSentence
from gensim.corpora.mmcorpus import MmCorpus

from pyLDAvis.gensim_models import prepare as ldavis_gensim_prepare
import pickle
from pathlib import Path
from typing import List, Generator



def construct_data_dict(prepped_data:str|Path, dict_save_path:str|Path, 
                        filter_no_below:int=5, filter_no_above:float=0.5):
    """build a data dictionary from streamed corpus

    Args:
        prepped_data (str | Path): data to build dictionary fed to Linesentence
        dict_save_path (str | Path): path to save dictionary
        filter_no_below (int, optional): minimum count of word to include in the dictionary.
                                            passed to dictionary.filter_extreems. Defaults to 5.
        filter_no_above (float, optional): max percent of documents a word must be in to be included. 
                                            passed to dictionary.filter_extreems. Defaults to 0.5.
    """
    
    docs = LineSentence(prepped_data)
    data_dict = Dictionary(docs)
    data_dict.filter_extremes(no_below=filter_no_below, 
                              no_above=filter_no_above)
    data_dict.compactify()
    data_dict.save(str(dict_save_path))



def doc_bow_generator(trigram_revs:str|Path, data_dict_path:str|Path) -> Generator[List, None, None]:
    """reads in text corpus and yields a doc2bow representation of each document

    Args:
        trigram_revs (str | Path): Path to text file
        data_dict_path (str | Path): path to built dictionary

    Yields:
        Generator[List[Tuple(int, int)], None, None]: document in doc2bow format
    """
    data_dict = Dictionary.load(str(data_dict_path), mmap='r')
    docs = LineSentence(trigram_revs)
    for doc in docs:
        yield data_dict.doc2bow(doc)

def build_mm_corpus(trigram_docs_bow_path:str|Path, trigram_revs_path:str|Path, data_dict_path:str|Path):
    """builds and serialized documents in Matrix Market format using Mmcorpus

    Args:
        trigram_docs_bow_path (str | Path): Path to save Mmcorpus
        trigram_revs_path (str | Path): path to text file to transform (passed to doc_bow_generator)
        data_dict_path (str | Path): path to built dictionary (passed to doc_bow_generator)
    """
    MmCorpus.serialize(str(trigram_docs_bow_path), doc_bow_generator(trigram_revs_path, data_dict_path))


def build_lda_model(trigram_docs_bow_path:str|Path, data_dict_path:str|Path, lda_model_save_path:str|Path, 
                    num_topics:int, passes:int, iterations:int, eval_every:int|None=None):
    """trains an lda model via LdaMulticore.

    Args:
        trigram_docs_bow_path (str | Path): Path to Mmcorpus file
        data_dict_path (str | Path): Path to data dictionary
        lda_model_save_path (str | Path): save path for lda model
        num_topics (int): number of topic to find
        passes (int): number of passes over corpus
        iterations (int): iterations over the doc chunks
        eval_every (int | None, optional): estimate log perplexity after this many steps. Defaults to None.
    """
    trigram_docs_bow = MmCorpus(str(trigram_docs_bow_path))
    trigram_dict = Dictionary.load(str(data_dict_path), mmap='r')

    lda_model = LdaMulticore(corpus=trigram_docs_bow,
                            id2word=trigram_dict,
                            num_topics=num_topics,
                            passes=passes,
                            iterations=iterations,
                            eval_every=eval_every)
    
    lda_model.save(str(lda_model_save_path))


def prep_ldavis(model_path:str|Path, corpus_path:str|Path, trigram_dict_path:str|Path, prepped_save_path:str|Path):
    """prepare and save pyLDAvis visualization

    Args:
        model_path (str | Path): path to lda model
        corpus_path (str | Path): path to Mmcorpus
        trigram_dict_path (str | Path): Path to dictionary
        prepped_save_path (str | Path): save path for visualization
    """
    trigram_docs_bow = MmCorpus(str(corpus_path))
    trigram_dict = Dictionary.load(str(trigram_dict_path), mmap='r')
    lda_model = LdaMulticore.load(str(model_path), mmap='r')

    lda_vis_prepped = ldavis_gensim_prepare(lda_model, trigram_docs_bow, trigram_dict)

    with open(prepped_save_path, 'wb') as prepped_path:
        pickle.dump(lda_vis_prepped, prepped_path)





if __name__ == '__main__':
    from utils import load_config
    from argparse import ArgumentParser
    from datetime import datetime

    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to config.yaml', type=str)
    args = parser.parse_args()

    # load config
    config = load_config(args.config_path)


    # create paths
    base_data_folder = Path(config['paths']['base_data_folder'])

    trigram_reviews = base_data_folder / 'trigram_reviews.txt'
    trigram_dict = base_data_folder / 'trigram_dict.dict'
    mm_corpus = base_data_folder / 'trigram_doc_bow.mmcorp'


    base_model_path = base_data_folder / 'models'
    lda_model_folder = base_model_path / 'lda_model'
    lda_model_folder.mkdir(parents=True, exist_ok=True)

    lda_model_path = lda_model_folder / 'lda_model.mod'


    lda_vis_path = base_model_path / 'lda_vis_prep.pkl'

    start = datetime.now()
    print('Building Data Dictionary...')
    # create and save the data dictionary
    construct_data_dict(prepped_data=trigram_reviews, 
                        dict_save_path=trigram_dict,
                        filter_no_below=config['lda']['filter']['no_below'],
                        filter_no_above=config['lda']['filter']['no_above'])
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nBuilding Matrix Market Corpus...')
    # create and save Mmcorpus
    build_mm_corpus(trigram_docs_bow_path=mm_corpus, 
                    trigram_revs_path=trigram_reviews,
                    data_dict_path=trigram_dict)
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nTraining LDA Model...')
    # train lda model
    build_lda_model(trigram_docs_bow_path=mm_corpus,
                    data_dict_path=trigram_dict,
                    lda_model_save_path=lda_model_path,
                    num_topics=config['lda']['num_topics'],
                    passes=config['lda']['passes'],
                    iterations=config['lda']['iterations'])
    print('Duration: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nPrepping LDA Visualization...')
    # create lda visualization
    prep_ldavis(model_path=lda_model_path,
                corpus_path=mm_corpus,
                trigram_dict_path=trigram_dict,
                prepped_save_path=lda_vis_path)
    print('Duration: ', str(datetime.now() - start))