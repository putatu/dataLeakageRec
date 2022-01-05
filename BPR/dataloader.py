
def load_data(path, test_year, num_years_added, mode):
    
    """
    test_year: either year 5 or year 7 in our paper
    num_years_added: number of years of future data added
    mode: either train or test mode
    """

    if mode == "train":
        data_filename = path+'data.data.RATING'
        train_filename = path+'train.train.RATING'
        test_filename = path+'test.train.RATING'
    elif mode == "test":
        data_filename = path+'data.data.RATING'
        train_filename = path+'train.test.RATING'
        test_filename = path+'test.test.RATING'     

    num_ratings = 0
    
    train = {}
    test = []
    available_items = set()
    num_ratings = 0

    num_user = 0
    num_item = 0

    with open(data_filename, 'r') as f:
        for line in f:
            arr = line.split('\t')
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4])

            if (year < test_year) or (year > test_year and year <= test_year+num_years_added):
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
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
                if user in train:
                    train[user].append(item)
                else:
                    train[user] = [item]
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    num_user += 1
    num_item +=1
    
    test = []
    with open(test_filename, 'r') as f:
        for line in f:
            arr = line.split('\t')
            user, item, year= int(arr[0]), int(arr[1]), int(arr[4]) 
            if (year  == test_year):
                test.append([user, item, year])

    return train, test, available_items, num_user, num_item, num_ratings



