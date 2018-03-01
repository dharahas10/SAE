import os
import pickle
import time
from pprint import pprint

import numpy as np

from src.utils.helper import duration


class SingleRatingDataLoader():

    def __init__(self):
        np.random.seed(2156)

        self._train = {}
        self._test = {}

        self._userHash = {}
        self._userCounter = 1
        self._itemHash = {}
        self._itemCounter = 1
    
    
    def _getUserIndex(self, userId):
        hash_id = hash(userId)
        if hash_id not in self._userHash:
            self._userHash[hash_id] = self._userCounter
            self._userCounter += 1
        return self._userHash[hash_id]


    def _getItemIndex(self, itemId):
        hash_id = hash(itemId)
        if hash_id not in self._itemHash:
            self._itemHash[hash_id] = self._itemCounter
            self._itemCounter += 1
        return self._itemHash[hash_id]
    

    def _sortByASC(self, data):
        for _, val in data.items():
            items, ratings = zip(*sorted(zip(val['items'], val['ratings'])))
            val['items'] = list(items)
            val['ratings'] = list(ratings)
    

    def _appendOneRating(self, userIndex, itemIndex, rating, data):
        if userIndex not in data:
            data[userIndex] = {
                'ratings': [],
                'items': []
            }
        
        data[userIndex]['items'].append(itemIndex)
        data[userIndex]['ratings'].append(rating)
        

    def _appendRating(self, userIndex, itemIndex, rating):

        if np.random.uniform() <= self._splitRatio:
            self._appendOneRating(userIndex, itemIndex, rating, self._train)
        else:
            self._appendOneRating(userIndex, itemIndex, rating, self._test)
    

    def _loadRatings(self, config):

        filename = os.path.join(config['path'], config['file_name'])
        assert os.path.isfile(filename), "ERROR:: Dataset file not found"

        delimiter = config['delimiter']
        with open(filename) as file:
            for line in file:
                userId, itemId, rating, *_ = line.split(sep=delimiter)
                rating = float(rating)
                
                userIndex = self._getUserIndex(userId)
                itemIndex = self._getItemIndex(itemId)

                self._appendRating(userIndex, itemIndex, rating)

                
    def _reorderIndices(self):
        self._sortByASC(self._train)
        self._sortByASC(self._test)


    def _computeRatingsPerUser(self, data):
        for key, val in data.items():
            assert len(val['items']) == len(val['ratings']),\
             "ERROR:: Number of ratings and indices are not matched for user-counter: {}".format(key)
            
            data[key]['nRatings'] = len(val['ratings'])


    def _computeNoOfRatings(self):
        self._computeRatingsPerUser(self._train)
        self._computeRatingsPerUser(self._test)


    def _removeTestOnlyItems(self):
        testItems = set()
        trainItems = set()

        for _, user in self._test.items():
            testItems.update(user['items'])
        
        for _, user in self._train.items():
            trainItems.update(user['items'])
        
        self._nItems = len(trainItems)
        
        testOnlyItems = list(testItems - (testItems & trainItems))

        if len(testOnlyItems) != 0:
            print("\n-------> Found items only present in test dataset: {} and are {}".format(len(testOnlyItems), testOnlyItems))
            for item in testOnlyItems:
                for _, user in self._test.items():
                    if item in user['items']:
                        idx = user['items'].index(item)
                        del user['items'][idx]
                        del user['ratings'][idx]
            print("--------> Remove items present only in test dataset")                



        
        

    def _getTotalRatings(self, data):
        ratings = 0
        for _, val in data.items():
            if val is not None:
                ratings += val['nRatings']
        
        return ratings

    def _computeInfo(self):

        # Calculating U/V type no of users and items
        self._nUsersTrain = len(self._train.keys())
        self._nUsersTest = len(self._test.keys())
        
        self._nTrainRatings = 0
        for _, user in self._train.items():
            self._nTrainRatings += user['nRatings']
        
        self._nTestRatings = 0
        for _, user in self._test.items():
            self._nTestRatings += user['nRatings']

        self._nRatings = self._nTrainRatings + self._nTestRatings
        self._sparsity = 100 - (self._nRatings/(self._nUsersTrain*self._nItems))


    def _checkAndComputeInfo(self):
        # Removing items only present in the test dataset
        self._removeTestOnlyItems()
        # Compute no of ratings for each user in train and test
        self._computeNoOfRatings()
        # Compute Info 
        self._computeInfo()
        


    def convertAndSave(self, config):

        start = time.time()

        self._splitRatio = config['split_ratio']
        # appending ratings to self._train and self._test
        self._loadRatings(config)
        # sorting each users ratings according to asc of their items
        self._reorderIndices()
        # Trimming dataset from anamolies and info structure
        self._checkAndComputeInfo()
        # save to pickle file
        self._save(config['out'])

        data = {
            'train': self._train,
            'test': self._test,
            'info': {
                'total_ratings': self._nRatings,
                'train_ratings': self._nTrainRatings,
                'test_ratings': self._nTestRatings,
                'nUsers': self._nUsersTrain,
                'nItems': self._nItems,
                'nTestUsers': self._nUsersTest,
                'sparsity': self._sparsity
            }
        }

        stop = time.time()
        print("\nTime to proccess the dataset: {}".format(duration(start, stop)))
        return data
    
    def _save(self, filename):

        data = {
            'train': self._train,
            'test': self._test,
            'info': {
                'total_ratings': self._nRatings,
                'train_ratings': self._nTrainRatings,
                'test_ratings': self._nTestRatings,
                'nUsers': self._nUsersTrain,
                'nItems': self._nItems,
                'nTestUsers': self._nUsersTest,
                'sparsity': self._sparsity
            }
        }
        with open(filename, 'wb') as output:
            pickle.dump(data, output)
        
        print("Saved Succesfully to location: {}".format(filename))