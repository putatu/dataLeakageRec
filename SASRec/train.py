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
parser.add_argument('--maxlen', default = 10, help = 'maximum sequence length', type = int)
parser.add_argument('--lr', default = 0.0001, help = 'maximum sequence length', type = float)
parser.add_argument('--dropout_rate', default = 0.2, type = float)
parser.add_argument('--l2_emb', default = 0.0, help ='regularization coeficient', type = float)
parser.add_argument('--hidden_units', default = 64, type = int)
parser.add_argument('--num_blocks', default = 1, help ='number of blocks', type = int)






args = parser.parse_args()

dataset_name = args.data
mode = "train"
device = 'cuda:'+args.gpu
test_year = args.test_year
num_years_added = args.num_years_added
data_path = args.data_path+dataset_name+'/'

dataset = data_partition(data_path, mode, test_year, num_years_added)
[user_train, user_test, usernum, itemnum, available_items, num_ratings] = dataset

print("Number of users: ", usernum)
print("Number of items: ", itemnum)
print("Number of test users: ", len(user_test))
print("Number of ratings: ", num_ratings)
print("Number of available items: ", len(available_items))


batch_size = 128


maxlen = args.maxlen
lr = args.lr
dropout_rate = args.dropout_rate
l2_emb = args.l2_emb
hidden_units = args.hidden_units
num_blocks = args.num_blocks

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
for epoch in range(epoch_start_idx, num_epochs + 1):
	t0 = time.time()    
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
	print(time.time() - t0)
	print("Dataset {} Test_year {}, num_years {} loss in epoch {}: {}, in time {}".format(dataset_name, test_year, num_years_added, epoch, loss.item(), time.time() - t0)) 
	if epoch % display_step == 0:
		model.eval()
		t1 = time.time()
		k = 20
		print('Evaluating', end='')        
		t_valid = evaluate_valid(model, dataset, maxlen, k)
		hr = t_valid[1]
		ndcg = t_valid[0]
		print("Epoch: ", epoch, "HR: ", hr, "NDCG: ", ndcg)
		model.train()
