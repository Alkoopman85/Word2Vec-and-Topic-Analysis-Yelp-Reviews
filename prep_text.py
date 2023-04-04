import spacy
import json
from gensim.models.phrases import Phrases, FrozenPhrases, ENGLISH_CONNECTOR_WORDS
from gensim.models.word2vec import LineSentence
from typing import Generator, Any
from pathlib import Path


def load_business_ids(business_ids_file_path:str|Path|None) -> dict[str, int] | None:
    """takes a text file, and creates a dictionarry that acts as a quick search/lookup,
        all lines in the file should be unique

    Args:
        business_ids_file_path (str | Path | None): path to file containing ids

    Returns:
        dict[str, int] | None: returns a dictionarry containing the ids as keys, with arbitrary values
    """
    try:
        with open(business_ids_file_path, 'r', encoding='utf-8') as bus_ids:
            business_ids = {id_.rstrip('\n').strip(): 1 for id_ in bus_ids}
    except TypeError:
        business_ids = None
    return business_ids

def checker(to_check:str, check_dict:dict) -> bool:
    """checks if a key is in a dictionarry. catches the Exception,
        and returns True if the key is present and False if it is not.

    Args:
        to_check (str): key to find
        check_dict (dict): dictionary of of wich to search

    Returns:
        bool: True if the key is present and False if it is not
    """
    try:
        _ = check_dict[to_check]
        return True
    except KeyError:
        return False
    
def get_json_line(file_path:str|Path, business_ids:dict|None=None) -> Generator[Any, None, None]:
    """reads in a json file yields a string, 
        Note: json file must have "business_id" and "text" as keys

    Args:
        file_path (str | Path): path to json file
        business_ids (dict | None, optional): lookup table for ids. Defaults to None.

    Yields:
        Generator[Any, None, None]: yields the text string for that line of the json file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_string = json.loads(line)
            if business_ids is not None:
                if checker(json_string['business_id'], business_ids):
                    yield json_string['text']
                else:
                    continue
            else:
                yield json_string['text']


def write_text_file_from_json(infile:str|Path, outfile:str|Path, business_ids:dict|None=None) -> None:
    """write out text to file one review per line.

    Args:
        infile (str | Path): path to json file
        outfile (str | Path): path to text file to write output
        business_ids (dict | None, optional): lookup table for ids.. Defaults to None.
    """
    review_count = 0
    with open(outfile, 'w', encoding='utf-8') as out_file:
        for line in get_json_line(file_path=infile, business_ids=business_ids):
            out_file.write(line.replace('(', '').replace(')', '').replace('\n', '\\n') + '\n')
            review_count += 1
    print(f'{review_count:,} reviews written to file')



def include(token) -> bool:
    """ckecks if a spacy token should be included when iterating

    Args:
        token: spacy token as returned by nlp=spacy.load(); nlp(text)[0]

    Returns:
        bool: True if token is not punctuation or space else False
    """
    return not token.is_punct and not token.is_space and not token.is_digit

def get_reviews_generator(review_txt_file:str|Path) -> Generator[str, None, None]:
    """read in a text file and return a string

    Args:
        review_txt_file (str | Path): path to text file

    Yields:
        Generator[str, None, None]: text string
    """
    with open(review_txt_file, 'r', encoding='utf-8') as reviews_file:
        for review in reviews_file:
            yield review.replace('\\n', '\n')


def write_unigram_sentences_and_reviews(text_infile:str|Path, sentence_outfile:str|Path, unigram_review_outfile:str|Path, 
                                        spacy_model, batch_size:int=10000, n_process:int=4) -> None:
    """reads in a text file, each line processed through spacy model,
        unigram sentences written to file one sentence per line.
        unigram reviews written to file one review per line.


    Args:
        text_infile (str | Path): Path to corpus to process
        sentence_outfile (str | Path): output file path for unigram sentences
        unigram_review_outfile (str | Path): output file path for unigram reviews
        spacy_model: spacy model for processing
        batch_size (int, optional): batchsize to pass to spacy_model.pipe(). Defaults to 10000.
        n_process (int, optional): n_processes to pass to spacy_model.pipe(). Defaults to 4.
    """
    total_sentences = 0
    total_reviews = 0
    with open(sentence_outfile, 'w', encoding='utf-8') as sent_outfile:
        with open(unigram_review_outfile, 'w', encoding='utf-8') as review_outfile:
            for review in spacy_model.pipe(get_reviews_generator(text_infile), batch_size=batch_size, n_process=n_process):
                if review and review.text != '\n' and review.text != '':
                    review_outfile.write(' '.join([token.lemma_ for token in review if include(token)]) + '\n')
                    total_reviews += 1
                    for sentence in review.sents:
                        if str(sentence) != '' and str(sentence) != ' ':
                            sent_outfile.write(' '.join([token.lemma_ for token in sentence if include(token)]) + '\n')
                            total_sentences += 1
    print(f'Sentences written to file: {total_sentences:,}')


def train_save_phraser(source_sent_file_path:str|Path, dest_sent_file_path:str|Path, phraser_save_path:str|Path) -> FrozenPhrases:
    """trains a gensim Phrases model and saves it. then writes the phrases (bigram, trigram, etc.) transformed data to a file

    Args:
        source_sent_file_path (str | Path): path to file passed to gensim Linesentence
        dest_sent_file_path (str | Path): path to file for writing transformed text 
        phraser_save_path (str | Path): path to save phraser model

    Returns:
        FrozenPhrases: phraser model
    """
    sents = LineSentence(source_sent_file_path)
    phraser = Phrases(sents, connector_words=ENGLISH_CONNECTOR_WORDS)
    phraser.save(str(phraser_save_path))
    phraser = phraser.freeze()
    with open(dest_sent_file_path, 'w', encoding='utf-8') as file:
        for sent in sents:
            gram_sent = phraser[sent]
            file.write(' '.join(gram_sent) + '\n')
    return phraser


def create_trigram_reviews_for_lda(unigram_reviews_path:str|Path, transformed_reviews_save_path:str|Path, bigram_phraser_path:str|Path, 
                                   trigram_phraser_path:str|Path, stopwords:set|frozenset|None=None) -> None:
    """reads in a file of unigram text, and saves a file of trigram text.

    Args:
        unigram_reviews_path (str | Path): Path to unigram text file
        transformed_reviews_save_path (str | Path): output save path for transformed text
        bigram_phraser_path (str | Path): Path to saved bigram Phrases model
        trigram_phraser_path (str | Path): Path to saved trigram Phrases model
        stopwords (set | frozenset | None, optional): stopwords to filter out. Defaults to None.
    """
    
    stopwords_dict = dict()
    if stopwords is not None:
        stopwords_dict = {word: 1 for word in stopwords}

    bigram_phraser = FrozenPhrases.load(str(bigram_phraser_path), mmap='r')
    trigram_phraser = FrozenPhrases.load(str(trigram_phraser_path), mmap='r')

    in_revews = LineSentence(unigram_reviews_path)
    with open(transformed_reviews_save_path, 'w', encoding='utf-8') as reviews_write_file:
        for unigram_rev_normalized in in_revews:

            bigram_rev_normalized = bigram_phraser[unigram_rev_normalized]
            trigram_rev_normalized = trigram_phraser[bigram_rev_normalized]


            if stopwords is not None:
                trigram_rev_normalized = [token for token in trigram_rev_normalized if not checker(token, stopwords_dict)]

            
            
            reviews_write_file.write(' '.join(trigram_rev_normalized) + '\n')



if __name__ == '__main__':
    from utils import load_config
    from argparse import ArgumentParser
    from datetime import datetime
    
    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to config.yaml', type=str)
    args = parser.parse_args()

    # load config file
    config = load_config(args.config_path)

    # load spacy_model
    spacy_model = spacy.load(config['data_prep']['spacy_model_string'])

    # create paths for files
    business_ids_path = config['paths']['business_ids']
    yelp_reviews_dataset_path = config['paths']['yelp_reviews_dataset_path']

    base_data_folder = Path(config['paths']['base_data_folder'])
    raw_reviews = base_data_folder / 'raw_reviews.txt'
    unigram_sents = base_data_folder / 'unigram_sents.txt'
    unigram_reviews = base_data_folder / 'unigram_reviews.txt'
    bigram_sents = base_data_folder / 'bigram_sents.txt'
    trigram_sents = base_data_folder / 'trigram_sents.txt'
    trigram_reviews = base_data_folder / 'trigram_reviews.txt'
    
    # phraser_paths
    base_model_path = base_data_folder / 'models'
    base_model_path.mkdir(parents=True, exist_ok=True)
    bigram_phraser_path = base_model_path / 'bigram_phraser'
    trigram_phraser_path = base_model_path / 'trigram_phraser'
    
    # load business ids to include
    business_ids = load_business_ids(business_ids_path)

    start = datetime.now()
    print('Wrighting Raw Reviews...')
    # write out raw reviews extracted from json file
    write_text_file_from_json(infile=yelp_reviews_dataset_path, 
                            outfile=raw_reviews,
                            business_ids=business_ids)
    print('Durrarion: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nWrighting Unigram Sentences/Reviews...')
    # write out prcessed unigram sentences and reviews to files
    write_unigram_sentences_and_reviews(text_infile=raw_reviews,
                                        sentence_outfile=unigram_sents,
                                        unigram_review_outfile=unigram_reviews,
                                        spacy_model=spacy_model,
                                        batch_size=config['data_prep']['batch_size'],
                                        n_process=config['data_prep']['n_process'])
    print('Durrarion: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nWrighting Bigram Sentences and Saving Phraser Model...')
    # train bigram Phrases model and write out bigram sentences
    train_save_phraser(source_sent_file_path=unigram_sents,
                       dest_sent_file_path=bigram_sents,
                       phraser_save_path=bigram_phraser_path)
    print('Durrarion: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nWrighting Trigram Sentences and Saving Phraser Model...')
    # train trigram Phrases model and write out trigram sentences
    train_save_phraser(source_sent_file_path=bigram_sents,
                       dest_sent_file_path=trigram_sents,
                       phraser_save_path=trigram_phraser_path)
    print('Durrarion: ', str(datetime.now() - start))

    start = datetime.now()
    print('\nWrighting Trigram Reviews...')
    # transform and write out trigram reviews file
    create_trigram_reviews_for_lda(unigram_reviews_path=unigram_reviews,
                                    transformed_reviews_save_path=trigram_reviews,
                                    bigram_phraser_path=bigram_phraser_path,
                                    trigram_phraser_path=trigram_phraser_path,
                                    stopwords=spacy_model.Defaults.stop_words if config['data_prep']['remove_stop'] else None)
    print('Durrarion: ', str(datetime.now() - start))

