import numpy as np
import random
import pandas as pd
import os
from torch_geometric.utils import remove_self_loops, subgraph, from_scipy_sparse_matrix
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import itertools
import sys

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def build_relationship(x, thresh=0.25):
    #    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    #    df_euclid = df_euclid.to_numpy()
    #    np.save('credit_normal.npy', df_euclid)
    df_euclid = np.load('credit_normal.npy')
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id[:200]:
            if neig != ind:
                idx_map.append([ind, neig])
    print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def save_list(list_l, filename):
	# save idx
	with open(filename, 'w') as fp:
		for item in list_l:
			# write each item on a new line
			fp.write("%s\n" % item)
		print('Done')

def read_list(list_l, file_name):
	with open(file_name, 'r') as fp:
		for line in fp:
			# remove linebreak
			# linebreak is the last character of each line
			x = line[:-1]

			# add current item to the list
			list_l.append(x)
	list_l = list(map(int, list_l))
	# print("list_l", list_l)
	return list_l

# # Modified for subgraph. Train = 1500, test = 1000, val = 500
def read_credit_dataset(path, label_number=1500, sens_attr="Age", predict_attr="NoDefaultNextMonth"):
	dataset = "credit"
	# print('Loading {} dataset from {}'.format(dataset, path))
	idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
	header = list(idx_features_labels.columns)
	header.remove(predict_attr)
	header.remove('Single')

	#    # Normalize MaxBillAmountOverLast6Months
	#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
	#
	#    # Normalize MaxPaymentAmountOverLast6Months
	#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
	#
	#    # Normalize MostRecentBillAmount
	#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
	#
	#    # Normalize MostRecentPaymentAmount
	#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
	#
	#    # Normalize TotalMonthsOverdue
	#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

	# build relationship
	if os.path.exists(f'{path}/{dataset}_edges.txt'):
		edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
	else:
		edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
		np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

	features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
	labels = idx_features_labels[predict_attr].values
	idx = np.arange(features.shape[0])
	idx_map = {j: i for i, j in enumerate(idx)}
	edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
	adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
	                    dtype=np.float32)

	# build symmetric adjacency matrix
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	# features = normalize(features)
	adj = adj + sp.eye(adj.shape[0])
	# print("adj", adj)
	# print("adj", adj.shape)
	edge_index, edge_weight = from_scipy_sparse_matrix(adj)

	# print("edge_index", edge_index)
	# print(edge_index.shape)

	# adj = SparseTensor.from_edge_index(adj)
	# adj = adj.to_symmetric()

	features = torch.FloatTensor(np.array(features.todense()))
	labels = torch.LongTensor(labels)
	# print("labels", labels)
	# adj = sparse_mx_to_torch_sparse_tensor(adj)

	# random.seed(20)
	# label_idx_0 = np.where(labels == 0)[0]
	# label_idx_1 = np.where(labels == 1)[0]
	# random.shuffle(label_idx_0)
	# random.shuffle(label_idx_1)
	# print("len", len(label_idx_0))
	# print("len1", len(label_idx_1))
	#
	# num_train = 1500
	# num_test = 1000
	# num_val = 500

	# idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
	#                       label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
	# idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)): int(0.5 * len(label_idx_0)) + min(int(0.75 * len(label_idx_0)), num_val // 2)],
	#                     label_idx_1[int(0.5 * len(label_idx_1)): int(0.5 * len(label_idx_1)) + min(int(0.75 * len(label_idx_1)), num_val // 2)])
	# idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)): int(0.75 * len(label_idx_0)) + num_test // 2], label_idx_1[int(0.75 * len(label_idx_1)): int(0.75 * len(label_idx_1)) + num_test // 2])

	# save_list(idx_train, "idx_train.idx")
	# save_list(idx_val, "idx_val.idx")
	# save_list(idx_test, "idx_test.idx")

	idx_train = []
	idx_test = []
	idx_val = []

	idx_train = read_list(idx_train, path+"idx_train.idx")
	idx_val = read_list(idx_val, path+"idx_val.idx")
	idx_test = read_list(idx_test, path+"idx_test.idx")

	# print("train_idx", idx_train)
	# print("train_idx", len(idx_train))
	#
	# print("idx_val shape", len(idx_val))
	#
	# print("idx_test shape", len(idx_test))





	# print("pos train_idx", label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)].shape)
	# print("neg train idx", label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)].shape)
	#
	# print("pos val idx", label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))].shape)
	# print("neg val idx", label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))].shape)
	#
	# print("pos test idx", label_idx_0[int(0.75 * len(label_idx_0)):].shape)
	# print("neg test idx", label_idx_1[int(0.75 * len(label_idx_1)):].shape)

	all_idx = list(itertools.chain(idx_train, idx_val, idx_test))
	all_idx.sort()
	# print("all idx", len(all_idx))
	# print("all_idx", all_idx)
	# subgraph_edge_index
	edge_index, _ = subgraph(all_idx, edge_index, relabel_nodes=True)

	# print("subgraph_edge_index", edge_index.shape)
	# print("subgraph_edge_index", edge_index)

	labels = labels[all_idx]
	features = features[all_idx]

	# Since we have features and labels scliced appropriately, also, since I'm setting relabel_nodes = True (so we can have nodes 1-3K).
	# I need to modify

	# get index
	indices = [i for i in range(len(labels))]
	listidx_dataidx_dict = dict(zip(all_idx, indices))

	# print("listidx_dataidx_dict", listidx_dataidx_dict)


	train_mask = np.full_like(labels, False, dtype=bool)
	test_mask = np.full_like(labels, False, dtype=bool)
	val_mask = np.full_like(labels, False, dtype=bool)
	for i in all_idx:
		if i in idx_train:
			# print(listidx_dataidx_dict[i])
			train_mask[listidx_dataidx_dict[i]] = True
		elif i in idx_test:
			test_mask[listidx_dataidx_dict[i]] = True
		elif i in idx_val:
			val_mask[listidx_dataidx_dict[i]] = True


	# print("train_mask", list(train_mask))
	# print("test_mask", list(test_mask))
	# print("val_mask", list(val_mask))


	# train_mask = sample_mask(idx_train, labels.shape[0])
	# val_mask = sample_mask(idx_val, labels.shape[0])
	# test_mask = sample_mask(idx_test, labels.shape[0])
	train_mask = torch.tensor(train_mask)
	val_mask = torch.tensor(val_mask)
	test_mask = torch.tensor(test_mask)
	# print("train_mask", train_mask)

	for ind in idx_val:
		if ind in idx_train:
			print(ind)
		if ind in idx_test:
			print(ind)

	for ind in idx_test:
		if ind in idx_train:
			print(ind)

	#    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
	#    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
	#    idx_test = label_idx[int(0.75 * len(label_idx)):]

	sens = idx_features_labels[sens_attr].values.astype(int)
	sens = torch.FloatTensor(sens)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask,
	            val_mask=val_mask)
	# print("data", data)
	# print(data.x)
	return data
