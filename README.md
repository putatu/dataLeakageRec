# A Critical Study on Data Leakage in Recommender System Offline Evaluation
Code to reproduce the experiments from the paper: A critical study on data leakage in recommender system offline evaluation

This repository has the implementations for four models:
1. BPR
2. NeuMF: we follow https://github.com/hexiangnan/neural_collaborative_filtering
3. SASRec: we follow https://github.com/pmixer/SASRec.pytorch
4. LightGCN: we follow https://github.com/RUCAIBox/RecBole

# Environment Requirement
## BPR
- Tensorflow 1.14
- Python 3.6.9

## NeuMF
- Tensorflow 1.14
- Python 3.6.9
- Keras 2.3.0

## SASRec
- PyTorch >= 1.6

## LightGCN
- Install RecBole package in https://github.com/RUCAIBox/RecBole


# Dataset

1. Movielens-25m
2. Yelp
3. Amazon-music
4. Amazon-electronic

Datasets can be downloaded from 

# Data formart


User ID | Item Id | Rating | Timestamp | Year
--------|---------|--------|-----------|-----
........|.........|........|...........|.....



