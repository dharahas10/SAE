import json
import os
import pickle
from pprint import pprint

from Data.SingleRatingDataLoader import SingleRatingDataLoader


def main(config_file, config_dir):
    print('------------------- Dataset processing ---------------')
    
    config_loc = os.path.join(config_dir, config_file)

    # check config file
    assert os.path.isfile(config_loc), "ERROR:: No such {} config file is found".format(config_file)

    with open(config_loc, 'r') as jsonFile:
        config = json.load(jsonFile)
    
    print("\n--------------Current Dataset: {} --------------\n".format(config['name']))
    data_filename = config['out']

    if os.path.isfile(data_filename):
        print("INFO:: Found processed data")
        with open(config['out'], 'rb') as dataFile:
            data = pickle.load(dataFile)
            pprint(data['info'])
        
        return
    
    data = SingleRatingDataLoader().convertAndSave(config)
    pprint(data['info'])

    
if __name__ == "__main__":

    config_file = '20.json'
    config_dir = './config/Dataset/ml-10m'

    main(config_file, config_dir)
