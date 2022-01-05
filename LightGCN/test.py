#recbole: https://github.com/RUCAIBox/RecBole

from dataloader import load_data

import numpy as np
import argparse
import torch.optim as optim
import time
import torch
import time

from lightGCN import LightGCN

import heapq
from model.loss import RegLoss


import random

import pandas as pd



torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument('--test_year', type=int, default = 5, help='test_year')
parser.add_argument('--num_years_added', type=int, default = 5, help='num_years_added')

parser.add_argument('--data', type=str, default = 'ml')
parser.add_argument('--data_path', type=str, default = 'ml')

parser.add_argument('--gpu', default= '4', help='gpu')

args = parser.parse_args()

dataset_name = args.data

path = args.data_path+dataset_name+'/'

time_0 = time.time()


test_year = int(args.test_year) #test year: 5 or 7
num_years_added = int(args.num_years_added) #number of years of future data added
device = 'cuda:'+args.gpu



def getTopK(recommended, k, negative):
    preds = []
    recommended_num = 0
    item_num = 0
    while recommended_num < k:
        if recommended[item_num] in negative:
            preds.append(recommended[item_num])
            recommended_num+=1
        item_num+=1
    return preds


def getNDCG(k, preds, y):
    ndcg = 0
    for j in range(0, k):
        if preds[j] == y:
            ndcg = 1 / np.log2(j + 2)
    return ndcg


def get_train_batch(train, available_items):
    users = []
    poss = []
    negs = []
    for user in train:
        sampled_negatives = random.choices(list(available_items - set(train[user])), k=len(train[user]))
        for i in range(len(train[user])):
            users.append(user)
            poss.append(train[user][i])
            negs.append(sampled_negatives[i])
    return users, poss, negs
            


mode = "test"
train, data,test, available_items, num_user, num_item, num_ratings = load_data(path, test_year,num_years_added,  mode)
print("Number of users: ", num_user)
print("Number of items: ", num_item)
print("Number of test users: ", len(test))
print("Number of ratings: ", num_ratings)
print("Number of available items: ", len(available_items))


display_step = 20
num_epochs = 300


#get hyperparameters

'''
hypetermeters.txt

----------------------------------------------------------------------------------------------------------------------------------------------
dataset | test year | number of years of future data added | latent dimension | number of layers | regularation coefficients | learning rate |
-----------------------------------------------------------------------------------------------------------------------------------------------

'''


hyperparameters = {}
with open('hyperparameters.txt', 'r') as file:
    for line in file:
        arr = line.rstrip('\n').split('\t')
        hyperparameters[arr[0]+'_'+arr[1]+'_'+arr[2]] = [int(arr[3]), int(arr[4]), float(arr[5]), float(arr[6])]


key = dataset_name+'_'+str(test_year)+'_'+str(num_years_added)
latent_dim = hyperparameters[key][0]
n_layers = hyperparameters[key][1]
reg_weight = hyperparameters[key][2]
learning_rate = hyperparameters[key][3]


checkpoints_path = 'checkpoints/'+dataset_name+'_'+str(test_year)+'_'+str(num_years_added)+'.pt'
batch_size = 1024
model = LightGCN(data, num_user, num_item, latent_dim, n_layers, reg_weight, device).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)



best_loss = np.inf
for epoch in range(num_epochs):
    total_loss = []
    time_0 = time.time()
    users_set, poss_set, negs_set = get_train_batch(train,available_items)
    for s in range(np.int(num_ratings / batch_size)+1):
        user = torch.tensor(users_set[s*batch_size:(s+1)*batch_size],dtype=torch.long).to(device)
        pos = torch.tensor(poss_set[s*batch_size:(s+1)*batch_size],dtype=torch.long).to(device)
        neg = torch.tensor(negs_set[s*batch_size:(s+1)*batch_size],dtype=torch.long).to(device)
    
 
        optimizer.zero_grad()

        loss = model.calculate_loss(user, pos,neg)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    if np.mean(total_loss) < best_loss:
        best_loss = np.mean(total_loss)
        torch.save(model.state_dict(), checkpoints_path)

    print(epoch, np.mean(total_loss), time.time() - time_0,users_set[0], users_set[-1])
    
model = LightGCN(data, num_user, num_item, latent_dim, n_layers, reg_weight, device).to(device)
model.load_state_dict(torch.load(checkpoints_path))
model.eval()


HR = []
NDCG = []
k = 20
model.eval()
for test_i in range(0, len(test)):
    user_test = test[test_i][0]
    item_test = test[test_i][1]
    negative_items = list(available_items)
    scores = model.full_sort_predict(torch.tensor(user_test).to(device), torch.tensor(negative_items).to(device))

    scores = scores.detach().cpu().numpy()
    preds = dict(zip(negative_items, scores))
    recommended = heapq.nlargest(k, preds, key = preds.get)
    if item_test in recommended:
        HR.append(1)
    else:
        HR.append(0)

    NDCG.append(getNDCG(k, recommended, item_test)) 

hr = np.mean(HR)
ndcg = np.mean(NDCG)
print("Epoch: ", epoch, "HR: ", hr, "NDCG: ", ndcg)
