import os
import time
import torch
import argparse
import random
from model import SASRec
from tqdm import tqdm
from utils import *



parser = argparse.ArgumentParser()

parser.add_argument('--test_year', type=int, default = 5, help='test_year')
parser.add_argument('--num_years_added', type=int, default = 0, help='num_years_added')
parser.add_argument('--data', type=str, default = 'movielens')
parser.add_argument('--data_path', type=str, default = 'data/')
parser.add_argument('--gpu', default= '4', help='gpu')


args = parser.parse_args()

dataset_name = args.data
mode = "test"
device = 'cuda:'+args.gpu
test_year = args.test_year
num_years_added = args.num_years_added
data_path = args.data_path+dataset_name+'/'

dataset = data_partition(data_path, mode, test_year, num_years_added)
[user_train, user_test, usernum, itemnum, available_items, num_ratings] = dataset

print("Number of users: ", usernum)
print("Number of items: ", itemnum)
print("Number of ratings: ", num_ratings)
print("Number of test users: ", len(user_test))
print("Number of available items: ", len(available_items))


batch_size = 128

#get hyperparameters

'''
hypetermeters.txt

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dataset | test year | number of years of future data added | maximum seq length | hidden units| number of blocks | learning rate |
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''


hyperparameters = {}
with open('hyperparameters.txt', 'r') as file:
    for line in file:
        arr = line.rstrip('\n').split('\t')
        hyperparameters[arr[0]+'_'+arr[1]+'_'+arr[2]] = [int(arr[3]), int(arr[4]), int(arr[5]), float(arr[6])]


key = dataset_name+'_'+str(test_year)+'_'+str(num_years_added)


maxlen = hyperparameters[key][0]
hidden_units = hyperparameters[key][1]
num_blocks = hyperparameters[key][2]
lr = hyperparameters[key][3]

l2_emb = 0
dropout_rate = 0.2
num_heads = 1
num_batch = len(user_train) // batch_size
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)
model = SASRec(usernum, itemnum, hidden_units, maxlen, dropout_rate, num_heads, device, num_blocks).to(device) # no ReLU activation in original SASRec implementation?


for name, param in model.named_parameters():
	try:
		torch.nn.init.xavier_normal_(param.data)
	except:
		pass 
model.train()
epoch_start_idx = 1
bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

T = 0.0
num_epochs = 200
display_step = 20
best_loss = np.inf
checkpoints_path = 'checkpoints/'+dataset_name+'_'+str(test_year)+'_'+str(num_years_added)+'.pt'


for epoch in range(epoch_start_idx, num_epochs + 1):
	t0 = time.time()
	total_loss = []
	for step in range(num_batch):
		u, seq, pos, neg = sampler.next_batch()
		u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
		pos_logits, neg_logits = model(u, seq, pos, neg)
		pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
		adam_optimizer.zero_grad()
		indices = np.where(pos != 0)
		loss = bce_criterion(pos_logits[indices], pos_labels[indices])
		loss += bce_criterion(neg_logits[indices], neg_labels[indices])
		for param in model.item_emb.parameters(): loss += l2_emb * torch.norm(param)
		loss.backward()
		adam_optimizer.step()
		total_loss.append(loss.item())
	print(epoch, time.time() - t0,np.mean(total_loss), file=open('loss.txt', 'a'))
	if np.mean(total_loss) < best_loss:
		best_loss = np.mean(total_loss)
		torch.save(model.state_dict(), checkpoints_path)
model = SASRec(usernum, itemnum, hidden_units, maxlen, dropout_rate, num_heads, device, num_blocks).to(device) # no ReLU activation in original SASRec implementation?
model.load_state_dict(torch.load(checkpoints_path))
model.eval()	
print("Dataset {} Test_year {}, num_years {} loss in epoch {}: {}, in time {}".format(dataset_name, test_year, num_years_added, epoch, loss.item(), time.time() - t0)) 
k = 20
print('Evaluating', end='')


t_valid = evaluate_valid(model, dataset, maxlen, k)
hr = t_valid[1]
ndcg = t_valid[0]
print("Epoch: ", epoch, "HR: ", hr, "NDCG: ", ndcg)

