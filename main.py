import json
import os
import pickle
import time
from pprint import pprint

import tensorflow as tf

from src.helper import duration, normalize_data
from src.Trainer.MultiSAETrainer import MultiSAETrainer
from src.Trainer.SAETrainer import SAETrainer

if __name__ == "__main__":
    start = time.time()

    print("-------------------Stacked Auto-Encoder-------------------")
    # Configuration location of the Network
    curr_dataset = 'ml-100k'
    config_file = '5.json'
    config_dir = './config/Train/'+curr_dataset

    # Check for the config file
    assert os.path.isfile(os.path.join(
        config_dir, config_file)), "Configuartion file not found"

    with open(os.path.join(config_dir, config_file), 'r') as jsonFile:
        config = json.load(jsonFile)

    # print('\nCurrent Configuration of the Trainer : ')
    # pprint(config)

    # Checking processed data file
    assert os.path.isfile(config['dataset']), "Dataset not found"
    with open(config['dataset'], 'rb') as pcklFile:
        data = pickle.load(pcklFile)
        train, test, info = data['train'], data['test'], data['info']

    # # Dataset information
    # print('\nDataset Information :')
    # pprint(info)

    # normalise data
    train, test = normalize_data(train, test, normal=config['normalization'])

    # Trainer
    trainer = MultiSAETrainer(
    ) if config['isMulti'] else SAETrainer(config, info)
    trainer.execute(train, test)

    end = time.time()
    duration_str = duration(start, end)
    print("Time taken to complete (hh:mm:ss): {}".format(duration_str))
