
import scipy.sparse as sp
import numpy as np

class Dataset(object):

    def __init__(self, path,test_year, num_years_added, mode):
        if mode == "train":
            self.trainMatrix, self.trainList, self.available_items, self.num_ratings = self.load_rating_file_as_matrix(path+"data.data.RATING",path + "train.train.RATING", test_year, num_years_added)
            self.testRatings, self.testUsers = self.load_rating_file_as_list(path + "test.train.RATING",test_year)
            self.num_users, self.num_items = self.trainMatrix.shape
        elif mode == "test":
            self.trainMatrix, self.trainList, self.available_items, self.num_ratings = self.load_rating_file_as_matrix(path+"data.data.RATING",path + "train.test.RATING", test_year, num_years_added)
            self.testRatings, self.testUsers = self.load_rating_file_as_list(path + "test.test.RATING",test_year)
            self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename, test_year):
        ratingList = []
        testUsers = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                if (int(arr[-1]) == test_year):
                    user, item = int(arr[0]), int(arr[1])
                    ratingList.append([user, item])
                    testUsers.append(user)
                line = f.readline()
        return ratingList, testUsers
    
    def load_rating_file_as_matrix(self, filename_all, filename_train,test_year, num_years_added):
        num_users, num_items = 0, 0
        trainList = {}
        available_items = set()
        num_ratings = 0

        with open(filename_all, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                user, item, rating, year = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[-1])
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                line = f.readline()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename_all, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                user, item, rating, year = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[-1])
                if (int(arr[-1]) < test_year) or ((int(arr[-1]) > test_year) and (int(arr[-1]) <= test_year+num_years_added)):
                    
                    if (rating > 0):
                        num_ratings+=1
                        mat[user, item] = 1.0
                        if user in trainList:
                            trainList[user].append(item)
                        else:
                            trainList[user] = [item]
                        
                        available_items.add(item)
                line = f.readline()
        
        with open(filename_train, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                user, item, rating, year = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[-1])
                if (int(arr[-1]) == test_year):
                    if (rating > 0):
                        num_ratings+=1
                        mat[user, item] = 1.0
                        if user in trainList:
                            trainList[user].append(item)
                        else:
                            trainList[user] = [item]
                        
                        available_items.add(item)
                line = f.readline()    
        return mat, trainList, available_items, num_ratings