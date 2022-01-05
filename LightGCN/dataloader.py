
import numpy as np      
import random

def load_data(path,test_year, num_years_added, mode):

    if mode == "train":
        data_filename = path+'data.data.RATING'
        train_filename = path+'train.train.RATING'
        test_filename = path+'test.train.RATING'
    else:
        data_filename = path+'data.data.RATING'
        train_filename = path+'train.test.RATING'
        test_filename = path+'test.test.RATING'
    train = []  
    test = []
    num_ratings = 0
    num_item = 0
    import collections

    train = collections.defaultdict(list)
    available_items = set()
    num_ratings = 0
    num_item = 0
    num_user = 0
    train_data = []
    with open(data_filename, 'r') as f:
        for line in f:
            arr = line.split('\t')
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year < test_year) or (year > test_year and year <= test_year+num_years_added):
                train[user].append(item)

                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    with open(train_filename, 'r') as f:
        for line in f:
            arr = line.split('\t')
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])
            if (year  == test_year):

                num_ratings += 1
                train[user].append(item)

                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)


    num_user += 1
    num_item += 1
    
    test = []
    with open(test_filename, 'r') as f:
        for line in f:
            arr = line.split('\t')
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4]) 
            if (year  == test_year):
                test.append([user, item, year])

    data = np.zeros([num_user,num_item])
    for user in train:
        for item in train[user]:
            data[user, item] = 1

    return train, data, test, available_items, num_user, num_item, num_ratings

