from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from pathlib import Path
from datetime import datetime
import re





class FileManager:
    """
    file manager for saving models
    """
    def __init__(self, base_folder:str, file_prefix:str|None, keep_max:int|None=None, keyed_vectors_only:bool=False) -> None:
        """
        Args:
            base_folder (str): folder for saving the models
            file_prefix (str | None): first part of file and folder name. if None file_prefix='w2v_model'
            keep_max (int | None, optional): max number of models to keep, if None all saves are kept. Defaults to None.
            keyed_vectors_only (bool, optional): only save keyed vectors. Defaults to False.
                NOTE: if keyed_vectors_only is True no continued training is possible
        """
        self.base_folder = Path(base_folder)
        self.file_prefix = file_prefix if file_prefix is not None else 'w2v_model'
        self.keep_max = keep_max
        self.keyed_vectors_only = keyed_vectors_only

    def save(self, model, epoch:int) -> None:
        """saves the model/KeyedVectors

        Args:
            model (Word2Vec): gensim word2vec model
            epoch (int): current epoch
        """
        pathdir = self.base_folder / f'{self.file_prefix}_epoch_{epoch:0>4}'
        pathdir.mkdir(parents=True, exist_ok=True)
        if self.keyed_vectors_only:
            model.wv.save(str(pathdir / self.file_prefix))
        else:
            model.save(str(pathdir / self.file_prefix))
        self.check_del(epoch)

    def check_del(self, epoch:int) -> None:
        """keep self.keep_max most recent saves, deletes the rest

        Args:
            epoch (int): the current epoch
        """
        if self.keep_max is not None:
            if epoch <= self.keep_max:
                return
            else:
                max_to_del = epoch - self.keep_max
            for folder in self.base_folder.iterdir():
                file_epoch = re.search(r'_epoch_(\d+)', folder.name)
                if file_epoch:
                    epoch_num = int(file_epoch.groups()[0])
                else:
                    continue
                if epoch_num <= max_to_del:
                    for file in folder.iterdir():
                        file.unlink()
                    folder.rmdir()



class SaveVerboseModelCallback(CallbackAny2Vec):
    """callback that prints progress and saves the model using FileManager.
    """

    def __init__(self, base_folder:str, file_prefix:str|None=None, keep_max:int|None=None, keyed_vectors_only:bool=False) -> None:
        """
        Args:
            base_folder (str): folder for saving the models passed to FileManager
            file_prefix (str | None, optional): first part of file and folder name. passed to FileManager. Defaults to None.
            keep_max (int | None, optional): max number of models to keep, if None all saves are kept. passed to FileManager. Defaults to None.
            keyed_vectors_only (bool, optional): only save keyed vectors. passed to FileManager. Defaults to False.
                NOTE: if keyed_vectors_only is True no continued training is possible
        """

        self.file_manager = FileManager(base_folder, file_prefix, keep_max, keyed_vectors_only)
        self.epoch = 1


                
    def on_train_begin(self, model) -> None:
        """
            print message, get start time
        """
        print(f'Begin training over {model.epochs} epochs')
        self.train_begin = datetime.now()

    def on_epoch_begin(self, model) -> None:
        """
            print message, get epoch start time
        """
        print(f'Starting epoch: {self.epoch:>3} / {model.epochs:<3}', end='\r')
        self.epoch_begin = datetime.now()
    
    def on_epoch_end(self, model) -> None:
        """
            print message, and save
        """
        delta = (datetime.now() - self.epoch_begin)
        print(f'Finished epoch: {self.epoch:>3} / {model.epochs:<3}| epoch duration {str(delta)}')
        self.file_manager.save(model, self.epoch)
        self.epoch += 1
        

    def on_train_end(self, model) -> None:
        """
            print message
        """
        print(f'\nTraining Complete!, total duration: {str(datetime.now() - self.train_begin)}')
        


def train_word_2_vec(corpus_path:str|Path, epochs:int, vector_size:int, sg:int, hs:int, workers:int, window:int, min_count:int, 
                     model_checkpoint_folder:str|Path, base_prefix:str, keep_newest:int=3, keyed_vectors_only:bool=False):
    """train a gensim word2vec model

    Args:
        corpus_path (str | Path): path to the corpus file
        epochs (int): epochs to train over
        vector_size (int): the size of the vectors produced by the model
        sg (int): train a skipgram model 1, bag of words model 0
        hs (int): use hierarchical softmax 1, negative sampling 0
        workers (int): number of workers for word2vec model
        window (int): window size for word2vec model
        min_count (int): minimum word count freequency to include in model
        model_checkpoint_folder (str | Path): path to the folder for saving after each epoch
        base_prefix (str): first part of file and folder name when saving
        keep_newest (int, optional): max number of model saves to keep. Defaults to 3.
        keyed_vectors_only (bool, optional): only save keyed vectors not full model after each epoch. Defaults to False.
    """


    save_callback = SaveVerboseModelCallback(model_checkpoint_folder, base_prefix, keep_newest, keyed_vectors_only)

    Word2Vec(corpus_file=str(corpus_path), 
                vector_size=vector_size,
                sg=sg,
                hs=hs,
                min_count=min_count,
                workers=workers,
                window=window,
                epochs=epochs,
                callbacks=(save_callback,)
                )


if __name__ == '__main__':
    from utils import load_config
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path to config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    base_data_folder = Path(config['paths']['base_data_folder'])
    model_checkpoint_folder = base_data_folder / 'models/word2vec'
    model_checkpoint_folder.mkdir(parents=True, exist_ok=True)

    corpus_path = base_data_folder / 'trigram_sents.txt'

    train_word_2_vec(corpus_path=corpus_path,
                     epochs=config['word_2_vec']['epochs'],
                     vector_size=config['word_2_vec']['vector_size'],
                     sg=config['word_2_vec']['sg'],
                     hs=config['word_2_vec']['hs'],
                     workers=config['word_2_vec']['workers'],
                     window=config['word_2_vec']['window'],
                     min_count=config['word_2_vec']['min_count'],
                     model_checkpoint_folder=model_checkpoint_folder,
                     base_prefix=config['word_2_vec']['base_prefix'],
                     keep_newest=config['word_2_vec']['keep_newest'],
                     keyed_vectors_only=config['word_2_vec']['keyed_vectors_only']
                     )
