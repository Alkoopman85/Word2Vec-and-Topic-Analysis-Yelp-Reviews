import yaml
import numpy as np
from pathlib import Path


def load_config(config_path:str|Path)-> dict:
    """reads the config file and returns it as a dictionary

    Args:
        config_path (str | Path): Path to the yaml configuration file

    Returns:
        dict: dictionary representation of the configuration file
    """
    with open(config_path, 'r') as config_file:
        config_obj = yaml.safe_load(config_file)

    return config_obj

def get_file_length(file_name:str|Path) -> int:
    """runs through a file and counts the nuber of lines

    Args:
        file_name (str | Path): path to input file

    Returns:
        int: number of lines in a file
    """
    lines = 0
    with open(file_name, 'r') as file:
        for line in file:
            lines += 1
    return lines


def random_split_datafile(input_file:str|Path, train_file:str|Path, validation_file:str|Path, sample_size:int, test_ratio:float):
    """takes a random sample of lines in a file and writes them out to
        train and test sets. Note: the order of the sampled lines is 
        the same as they appear in the original file.

    Args:
        input_file (str | Path): Path to the file to sample from
        train_file (str | Path): Path to file for train set of lines
        validation_file (str | Path): path to file for test set of lines
        sample_size (int): the number of lines to sample (cannot exceed the total length of the input_file)
        test_ratio (float): the ratio of train to test sets
    """
    full_file_len = get_file_length(input_file)

    sample = np.random.choice(full_file_len, sample_size, False)

    test_size = int(sample_size * test_ratio)
    train, test = sample[test_size:], sample[:test_size]

    # turn into dict so lookup is O(1) instead of O(N)
    train_dict = {num: 1 for num in train}
    test_dict = {num: 1 for num in test}

    train_lines = 0
    test_lines = 0
    with open(input_file, 'r') as infile:
        with open(train_file, 'w') as train_file_out:
            with open(validation_file, 'w') as validation_file_out:

                for index, line in enumerate(infile):
                    # check if the index is part of the train set
                    try:
                        train_dict[index]
                        train_file_out.write(line)
                        train_lines += 1

                    except KeyError:
                        # check if the index is part of the test set
                        try:
                            test_dict[index]
                            validation_file_out.write(line)
                            test_lines += 1
                        except KeyError:
                            pass


    print(f'train lines written: {train_lines}, test lines written: {test_lines}')




def jaccard_distance(set_A:list|set, set_B:list|set) -> float:
    """Calculates the Jaccard distance between two sets

    Args:
        set_A (list | set): collection 1
        set_B (list | set): collection 2

    Returns:
        float: Jaccard Distance
    """
    union = len(set(set_A).union(set(set_B)))
    intersection = len(set(set_A).intersection(set(set_B)))

    j_distance = intersection / union

    return j_distance


