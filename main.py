import tensorflow as tf
import os
import json
import timeit
import pickle
from pprint import pprint

# imports models
from src.Trainer.SAETrainer import SAETrainer
from src.Trainer.MultiSAETrainer import MultiSAETrainer

if __name__ == "__main__":

    print("-------------------Stacked Auto-Encoder-------------------")
    # Configuration location of the Network
    curr_dataset = 'ml-100k'
    config_file = '1_U.json'
    config_dir = './config/Train/'+curr_dataset
    # Check for the config file
    assert os.path.isfile(os.path.join(config_dir, config_file)), \
                                        "Configuartion file not found"

    with open(os.path.join(config_dir, config_file), 'r') as jsonFile:
        config = json.load(jsonFile)

    print('\nCurrent Configuration of the Trainer : ')
    pprint(config)
    # Checking processed data file
    assert os.path.isfile(config['dataset']), "Dataset not found"
    with open(config['dataset'], 'rb') as pcklFile:
        data = pickle.load(pcklFile)
        train, test, info = data['train'], data['test'], data['info']

    # Dataset information
    print('\nDataset Information :')
    pprint(info)

    # Trainer
    trainer = MultiSAETrainer() if config['isMulti'] else SAETrainer(config, info)
    trainer.train(train)
