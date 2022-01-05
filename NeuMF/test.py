'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import heapq
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, multiply, Reshape, concatenate, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from dataset import Dataset
from time import time
import sys
import argparse
import random
import math
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--num_years_added', nargs='?', type=int, default= 0,
                        help='Number of years of future data added.')
    parser.add_argument('--test_year', nargs = '?', type =int, default = 5, help='selected year')
    parser.add_argument('--gpu', nargs='?', default = '5', help='gpu')
    parser.add_argument('--data', nargs='?',default='yelp', help='dataset')

    return parser.parse_args()



def init_normal(shape, name=None):
    return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0)


def gmf_get_model(num_users, num_items, latent_dim, regs):
    user_input = Input(shape=(1,), dtype='int32')
    item_input = Input(shape=(1,), dtype='int32')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim,
                                  embeddings_initializer = init_normal([num_users, latent_dim]), embeddings_regularizer = l2(regs), input_length=1,name = 'gmf_user_embedding')
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim,
                                  embeddings_initializer = init_normal([num_items, latent_dim]), embeddings_regularizer = l2(regs), input_length=1,name = 'gmf_item_embedding')   
    
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    predict_vector = multiply([user_latent, item_latent])
    
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "gmf_prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                output=prediction)

    return model
def mlp_get_model(num_users, num_items, layers = [20,10], reg = 0):
    num_layer = len(layers) #Number of layers in the MLP
    user_input = Input(shape=(1,), dtype='int32')
    item_input = Input(shape=(1,), dtype='int32')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), 
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1, name = 'mlp_user_embedding')
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2),
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1, name = 'mlp_item_embedding')
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    vector = concatenate([user_latent, item_latent])
    
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform',name = "mlp_prediction")(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def get_model(num_users, num_items, mf_dim, layers, reg):
    num_layer = len(layers) #Number of layers in the MLP
    user_input = Input(shape=(1,), dtype='int32')
    item_input = Input(shape=(1,), dtype='int32')
    


    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim,
                                  embeddings_initializer = init_normal([num_users, mf_dim]), embeddings_regularizer = l2(reg), input_length=1, name = 'neumf_gmf_user_embedding')
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim,
                                  embeddings_initializer = init_normal([num_items, mf_dim]), embeddings_regularizer = l2(reg), input_length=1, name = 'neumf_gmf_item_embedding')   
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2),
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1, name = 'neumf_mlp_user_embedding')
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2),
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1, name = 'neumf_mlp_item_embedding')

    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = multiply([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in np.arange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    gmf_user_embeddings = gmf_model.get_layer('gmf_user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('gmf_item_embedding').get_weights()
    model.get_layer('neumf_gmf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('neumf_gmf_item_embedding').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('mlp_user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('mlp_item_embedding').get_weights()
    model.get_layer('neumf_mlp_user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('neumf_mlp_item_embedding').set_weights(mlp_item_embeddings)
    
    for i in np.arange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
    gmf_prediction = gmf_model.get_layer('gmf_prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('mlp_prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in np.arange(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels



def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
data_path = args.data_path

test_year = args.test_year
dataset_name = args.data
num_years_added = args.num_years_added
mode = "test"
dataset = Dataset(data_path+dataset_name+'/', test_year, num_years_added, mode)
train, testRatings, available_items, trainList = dataset.trainMatrix, dataset.testRatings, dataset.available_items, dataset.trainList
num_users, num_items = train.shape
num_ratings = dataset.num_ratings

print("Number of users: ", num_users)
print("Number of items: ", num_items)
print("Number of ratings: ", num_ratings)
print("Number of test users: ", len(testRatings))
print("Number of available items: ", len(available_items))


#get hyperparameters

'''
hypetermeters.txt

----------------------------------------------------------------------------------------------------------------------------------------------
dataset | test year | number of years of future data added | latent dimension | number of negative samples | regularation coefficients | learning rate |
-----------------------------------------------------------------------------------------------------------------------------------------------

'''


hyperparameters = {}
with open('hyperparameters.txt', 'r') as file:
    for line in file:
        arr = line.rstrip('\n').split('\t')
        hyperparameters[arr[0]+'_'+arr[1]+'_'+arr[2]] = [int(arr[3]), int(arr[4]), float(arr[5]), float(arr[6])]


key = dataset_name+'_'+str(test_year)+'_'+str(num_years_added)

mf_dim = hyperparameters[key][0]
num_negatives = hyperparameters[key][1] #number of negative samples
regs = hyperparameters[key][2] #regularization coefficients
learning_rate = hyperparameters[key][3] #learning rate




num_epochs = 300
batch_size = 1024
layers = [64,32,16] + [mf_dim]
learner = 'adam'
verbose = 20
mf_pretrain = ''
mlp_pretrain = ''
        
model = get_model(num_users, num_items, mf_dim, layers, regs)
if learner.lower() == "adagrad": 
    model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
elif learner.lower() == "rmsprop":
    model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
elif learner.lower() == "adam":
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
else:
    model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

if mf_pretrain != '' and mlp_pretrain != '':
    gmf_model = gmf_get_model(num_users,num_items,mf_dim,regs)
    gmf_model.load_weights(mf_pretrain)
    mlp_model = mlp_get_model(num_users,num_items, layers, regs)
    mlp_model.load_weights(mlp_pretrain)
    model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
    print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))

checkpoints_path = 'checkpoints/'+dataset_name+'_'+str(test_year)+'_'+str(num_years_added)
best_loss = np.inf

for epoch in np.arange(1, num_epochs+1):
    time_0 = time()
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    hist = model.fit([np.array(user_input), np.array(item_input)], 
                     np.array(labels), 
                     batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
    loss = hist.history['loss'][0]

    if loss < best_loss:
        best_loss = loss
        model.save_weights(checkpoints_path, overwrite=True)
    print(time()-time_0)
model.load_weights(checkpoints_path)

hits = []
ndcgs = []
time_0 = time()
k = 20
for idx in range(len(testRatings)):
    user = testRatings[idx][0]
    gtItem = testRatings[idx][1]
    items = list(available_items)
    users = np.full(len(items), user, dtype = 'int32')
    
    predictions = model.predict([users, np.array(items)], 
                         batch_size=1024, verbose=0)
    predictions = np.squeeze(predictions)

    item_scores = dict(zip(items,predictions))
    recommended = heapq.nlargest(k, item_scores, key = item_scores.get)

    if gtItem in recommended:
        hits.append(1)
    else:
        hits.append(0)

    ndcgs.append(getNDCG(recommended, gtItem))

hr = np.mean(hits)
ndcg = np.mean(ndcgs)
print("Epoch: ", epoch, "HR: ", hr, "NDCG: ", ndcg)


