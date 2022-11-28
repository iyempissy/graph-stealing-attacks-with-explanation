#!/bin/bash
import argparse
import copy
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse, convert, from_networkx
from data_loader import load_data
from model import GCN, GCN_C, GCN_DAE, GCN_C_PyG, GCN_PyG
from utils import accuracy, get_random_mask, get_random_mask_ogb, nearest_neighbors, normalize
import networkx as nx
from sklearn.metrics import roc_curve, auc, average_precision_score
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
from fidelity import fidelity
from defenses import explanation_intersection
from scipy.stats import entropy

EOS = 1e-10

device_id = 1
if torch.cuda.is_available():
    torch.cuda.set_device(device_id)  # change this cos sometimes port 0 is full


# device_gnn_dae = torch.device('cuda:2')
# # device_gnn_c = torch.device("cuda:6")
#
# x = torch.tensor(1)
# x = x.to(device_gnn_dae)
# print("x")

def compute_resources(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Size: {:.3f} MB | ParamSize: {:.3f}'.format(size_all_mb, param_size))



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



class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    # load explanation model
    def load_model(self, path, model):
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(path))
        model.eval()
        return model



    # For getting classification loss for GCN_C
    def get_loss_learnable_adj(self, model, mask, features, labels, Adj, isPyG=True):
        # print("Features", features.shape) # 2708 x 1433
        # print("Adj", Adj.shape) # Dense matrix: 2708 x 2708

        # we need convert the adjacency to edge_index to use pyG
        if isPyG:
            Adj, edge_weight = dense_to_sparse(Adj)
            logits = model(features, Adj, edge_weight)
        else:

            logits = model(features, Adj)

        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_fixed_adj(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def get_loss_acc_normal(self, model, mask, features, edges, labels):
        logits = model(features, edges)
        logp = logits
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu


    def get_loss_adj(self, model, features, feat_ind):
        labels = features[:, feat_ind].float()
        new_features = copy.deepcopy(features)
        new_features[:, feat_ind] = torch.zeros(new_features[:, feat_ind].shape)
        logits = model(new_features)
        loss = F.binary_cross_entropy_with_logits(logits[:, feat_ind], labels, weight=labels + 1)
        return loss

    # for getting the loss of GCN_DAE after adding noise!
    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
        if ogb:  # if the feature values are not binary, then u can set ogb to true!
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
                masked_features = features + (noise * mask)

            logits, Adj = model(features, masked_features)
            indices = mask > 0

            if loss_t == 'bce':
                features_sign = torch.sign(features).cuda() * 0.5 + 0.5 #changes it to 0 and 1
                loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj = model(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj



    def train_test_normal(self, args):
        # For normal training and testing a GNN model
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, saved_model_path = load_data(args)
        print("original", original_adj.shape)
        G = nx.from_numpy_matrix(original_adj)
        data = from_networkx(G)
        print("data", data)
        edges = data.edge_index

        print("edges", edges)
        print(edges.shape)

        val_accuracies = []
        test_accuracies = []

        for trial in range(args.ntrials): #run for ntrials times

            model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                               num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                               sparse=args.sparse)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            bad_counter = 0
            best_val = 0
            best_model = None
            best_loss = 0
            best_train_loss = 0

            if torch.cuda.is_available():
                model = model.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                edges = edges.cuda()

            for epoch in range(1, args.epochs + 1):
                model.train()
                loss, accu = self.get_loss_acc_normal(model, train_mask, features, edges, labels)
                print("Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f}".format(epoch, loss.item(), accu))
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if epoch % 10 == 0:
                    model.eval()
                    val_loss, accu = self.get_loss_acc_normal(model, val_mask, features, edges, labels)
                    if accu > best_val:
                        bad_counter = 0
                        best_val = accu
                        best_model = copy.deepcopy(model)
                        best_loss = val_loss
                        best_train_loss = loss
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience:
                        break

            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(best_loss, best_val))
            best_model.eval()
            test_loss, test_accu = self.get_loss_acc_normal(best_model, test_mask, features,edges, labels)
            test_accuracies.append(test_accu.item())
            val_accuracies.append(best_val.item())
            print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accu))
        print("val_accuracies", val_accuracies)
        print("test_accuracies", test_accuracies)

        print("mean val_accuracies==>", np.mean(val_accuracies))
        print("std val_accuracies",np.std(val_accuracies))

        print("mean test_accuracies==>", np.mean(test_accuracies))
        print("std test_accuracies",np.std(test_accuracies))


        return val_accuracies, test_accuracies, best_model


    def run_fidelity(self, args):
        print("======================== Running fidelity===============================")
        if args.get_fidelity == 1:
            explanations, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, edge_index, \
            saved_model_path = load_data(args)
            
            print("Exp", explanations.shape)
            print("Feat", features.shape)
        else:
            raise ValueError("Set get_fildeity = 1 in the arguments")

        # GCN_C_PyG
        model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                           num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                           sparse=args.sparse)

        # # load saved explanation model!
        model = self.load_model(saved_model_path, model) #"./Cora_.pth.tar"
        print("new model", model)
        #[i for i in range(0, len(features))]
        fidelity_score_res = []
        print("len(features)", len(features))


        # compute sparsity / selected feature. This sparsity is for counting the number of 1
        # sum_each_exp_sparsity = torch.sum(explanations, 1)
        # print("sum_each_exp_sparsity", sum_each_exp_sparsity.shape)
        # print("sum_each_exp_sparsity", sum_each_exp_sparsity)
        # div_each_exp_sparsity = torch.div(sum_each_exp_sparsity, len(explanations[0])) #divide each by the feature_dim
        # print("div_each_exp_sparsity", div_each_exp_sparsity)
        # print("div_each_exp_sparsity", div_each_exp_sparsity.shape)
        # final_sparsity = torch.mean(div_each_exp_sparsity)
        #
        # print("final_sparsity", final_sparsity)

        sparsity_score_res = []

        for node_idx in range(0, len(features)):
            sparsity_score = entropy(explanations[node_idx])
            print("sparsity_score", sparsity_score)

            sparsity_score_res.append(sparsity_score)

            fidelity_score = fidelity(model, node_idx , features, edge_index=edge_index,  # the whole, so data.edge_index
                         node_mask=None,  # at least one of these three node, feature, edge
                         feature_mask=explanations[node_idx].reshape(1, -1),
                         edge_mask=None
                         )
            fidelity_score_res.append(fidelity_score)

        print("sparsity_score_res", sparsity_score_res)
        print("Final Sparsity score", sum(sparsity_score_res) / len(sparsity_score_res))
        print("Final Fidelity score", sum(fidelity_score_res) / len(fidelity_score_res))


    def run_intersection(self, args):
        original_exp, perturbed_exp = load_data(args)
        sparsity_score_res = []
        for node_idx in range(0, len(perturbed_exp)):
            sparsity_score = entropy(perturbed_exp[node_idx])

            sparsity_score_res.append(sparsity_score)

        print("Final Sparsity score", sum(sparsity_score_res) / len(sparsity_score_res))

        intersection = explanation_intersection(original_exp, perturbed_exp)
        print("Intersection", intersection)

    def pairwise_similarity(self, args):
        # For computing all pair feature similarity
        print("******* Using Pairwaise similarity*******")
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, \
        saved_model_path = load_data(args)

        similarity_matrix = cosine_similarity(features)

        avg_auroc = []
        avg_avg_prec = []

        for trial in range(args.ntrials):
            print("similarity_matrix.shape[0]", similarity_matrix.shape[0])
            print("int(similarity_matrix.shape[0] * 0.1)", int(similarity_matrix.shape[0] * 0.1))
            # #fixing seed
            # seed = args.seeds[trial]
            # np.random.seed(seed)
            # random.seed(seed)
            # print("idx_attack", idx_attack)

            if args.save_testset == 1:
                # save testset
                idx_attack = np.array(random.sample(range(similarity_matrix.shape[0]), int(similarity_matrix.shape[0] * 0.1))) #2708, 270
                save_list(idx_attack, "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(trial) + "_.idx")
            else:
                # load
                # The 100 runs experiment
                if args.run_all_testset == 1:
                    idx_attack_all = []
                    for k in range(args.ntrials):
                        idx_attack = []
                        idx_attack = read_list(idx_attack,
                                               "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(
                                                   k) + "_.idx")
                        idx_attack = np.array(idx_attack)
                        idx_attack_all.append(idx_attack)
                else:
                    # normal 10 runs experiment
                    idx_attack = []
                    idx_attack = read_list(idx_attack,
                                           "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(
                                               trial) + "_.idx")
                    idx_attack = np.array(idx_attack)

            # Do reconstruction metric
            if args.run_all_testset == 1:
                auroc, avg_prec = self.reconstruction_metric(original_adj, similarity_matrix, idx_attack_all, args.dataset,
                                                             trial, args.save_testset, args.run_all_testset, args.ntrials)
            else:
                auroc, avg_prec = self.reconstruction_metric(original_adj, similarity_matrix, idx_attack, args.dataset, trial, args.save_testset, args.run_all_testset, args.ntrials)

            avg_auroc.append(auroc)
            avg_avg_prec.append(avg_prec)

        print("===== Pairwise similarity=================")

        print("args.ntrials", args.ntrials)
        print("avg_auroc", avg_auroc)
        print("avg_avg_prec", avg_avg_prec)
        print("reconstructed auroc mean", np.mean(avg_auroc))
        print("reconstructed auroc std", np.std(avg_auroc))

        print("reconstructed avg_prec mean", np.mean(avg_avg_prec))
        print("reconstructed avg_prec std", np.std(avg_avg_prec))


        return auroc, avg_prec

    def train_end_to_end(self, args):
        # all_devices = args.devices
        # print("all_devices", all_devices)
        # all_devices = list(all_devices.split(","))
        # all_devices = list(map(int, all_devices))
        # print("all_devices", all_devices)
        #
        # print("cuda:"+str(all_devices[1]))
        # dev = "cuda:"+str(all_devices[1])
        # send some operation to a different GPU e.g for pubmed
        device_gnn_dae = torch.device(device_id) #cuda:1
        device_gnn_rec_dae = torch.device(device_id) #cuda:1
        # device_gnn_layer_1 = torch.device("cuda:3")

        if args.use_exp_as_reconstruction_loss == 1:
            explanations, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, \
            saved_model_path = load_data(args)
            print("Exp", explanations)
            print("Feat", features)
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, \
            saved_model_path = load_data(args)

        print("train_mask", train_mask) #True False ...
        print("test_mask", test_mask) #True False ...
        print("val_mask", val_mask) #True False ...

        print("train_mask", train_mask.shape) #2708 #sum(train_mask) 140
        print("test_mask", test_mask.shape) #2708 # sum(test_mask) 1000
        print("val_mask", val_mask.shape) #2708 # sum(val_mask) 500

        print("Original feature", features.shape) #([2708, 1433])
        print("Original Adj", original_adj.shape) #(2708, 2708)

        test_accu = []
        validation_accu = []
        added_edges_list = []
        removed_edges_list = []

        avg_auroc = []
        avg_avg_prec = []

        # Data fixed but model changes for 10 (trials) runs!
        for trial in range(args.ntrials): #run for ntrails times
            print("trial", trial)
            # model1 is DAE. For self supervision
            model1 = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj, nclasses=nfeats,
                             dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                             features=features.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                             non_linearity=args.non_linearity, normalization=args.normalization,
                             gen_mode=args.gen_mode, sparse=args.sparse)


            if args.use_exp_as_reconstruction_loss == 1:  # adding explanation reconstruction loss to the attack
                model_exp = GCN_DAE(nlayers=args.nlayers_adj, in_dim=nfeats, hidden_dim=args.hidden_adj,
                                    nclasses=nfeats,
                                    dropout=args.dropout1, dropout_adj=args.dropout_adj1,
                                    features=explanations.cpu(), k=args.k, knn_metric=args.knn_metric, i_=args.i,
                                    non_linearity=args.non_linearity, normalization=args.normalization,
                                    gen_mode=args.gen_mode, sparse=args.sparse)

            # # model2 is a 2-layer GCN. For classifier
            if args.dataset == "credit" or args.dataset == "pubmed":
                model2 = GCN_C(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                               num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                               sparse=args.sparse)
            else:
                model2 = GCN_C_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)

            if args.load_exp_model == 1:
                # load saved explanation model!
                model2 = self.load_model(saved_model_path, model2)
            # else retrain classification model from scratch!
            print("model2", model2)

            optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)

            if args.use_exp_as_reconstruction_loss == 1:
                optimizer_exp = torch.optim.Adam(model_exp.parameters(), lr=args.lr_adj, weight_decay=args.w_decay_adj)

            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model1 = model1.to(device_gnn_dae)#model1.cuda()  # move to a different GPU
                model2 = model2.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()

                if args.use_exp_as_reconstruction_loss == 1:
                    model_exp = model_exp.to(device_gnn_rec_dae)
                    explanations = explanations.to(device_gnn_rec_dae)

            best_val_accu = 0.0
            best_model2 = None
            best_Adj = None

            ''' Learning Adjacency '''
            for epoch in range(1, args.epochs_adj + 1):
                model1.train()
                if args.load_exp_model == 1:
                    # Since we assume that we have access to the trained model. Thus no need to retrain but only evaluate!
                    # when using trained explanation model
                    model2.eval()
                else:
                    model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if args.use_exp_as_reconstruction_loss == 1:
                    model_exp.train()
                    optimizer_exp.zero_grad()


                if args.dataset == "cora_ml" or args.dataset == "bitcoin" or args.dataset=="chameleon" or args.dataset=="credit" or args.dataset=="pubmed":
                    mask = get_random_mask_ogb(features, args.ratio).to(device_gnn_dae) # They all have the value of the 1/ratio
                    ogb = True #cos the feature values are floating point (may contain negatives) and not binary!
                else:
                    mask = get_random_mask(features, args.ratio, args.nr).cuda()
                    ogb = False

                # Evaluate after certain epochs and only compute a meaningful loss2 (classification) after certain epochs
                # i.e training only with DAE until certain epochs e.g 400 epochs
                if epoch < args.epochs_adj // args.epoch_d:
                    model2.eval()
                    loss1, Adj = self.get_loss_masked_features(model1.to(device_gnn_dae), features.to(device_gnn_dae), mask.to(device_gnn_dae), ogb, args.noise, args.loss)


                    if args.use_exp_as_reconstruction_loss == 1:
                        loss_exp, Adj_exp = self.get_loss_masked_features(model_exp.to(device_gnn_rec_dae), explanations.to(device_gnn_rec_dae), mask.to(device_gnn_rec_dae), ogb, args.noise, args.loss)
                        Adj = Adj.cuda() + Adj_exp.cuda()

                    loss2 = torch.tensor(0).cuda()
                else:
                    loss1, Adj = self.get_loss_masked_features(model1.to(device_gnn_dae), features.to(device_gnn_dae), mask.to(device_gnn_dae), ogb, args.noise, args.loss)

                    if args.use_exp_as_reconstruction_loss == 1:
                        loss_exp, Adj_exp = self.get_loss_masked_features(model_exp.to(device_gnn_rec_dae), explanations.to(device_gnn_rec_dae), mask.to(device_gnn_rec_dae),
                                                                          ogb, args.noise, args.loss)
                        Adj = Adj.cuda() + Adj_exp.cuda()

                    if args.dataset == "credit" or args.dataset == "pubmed":
                        loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features.cuda(), labels, Adj.cuda(),
                                                                  isPyG=False)
                    else:
                        loss2, accu = self.get_loss_learnable_adj(model2, train_mask, features.cuda(), labels, Adj.cuda(), isPyG=True)

                '''============ Final loss ===========>'''
                # loss1 = feature autoencoder noise
                # loss_exp = explanation autoencoder noise
                # loss2 = classification noise
                if args.use_exp_as_reconstruction_loss == 1:
                    loss = loss1.cuda() + loss_exp.cuda() + loss2
                else:
                    loss = loss1.cuda() * args.lambda_ + loss2

                loss.backward()
                optimizer1.step()
                optimizer2.step()

                if args.use_exp_as_reconstruction_loss == 1:
                    optimizer_exp.step()



                # print after every 100 epochs
                if epoch % 100 == 0:
                    print("Epoch {:05d} | Train Loss {:.4f}, {:.4f}".format(epoch, loss1.item() * args.lambda_,
                                                                            loss2.item()))

                # if epoch is greater than say 400, perform evaluation on the classification task
                if epoch >= args.epochs_adj // args.epoch_d and epoch % 1 == 0:
                    with torch.no_grad():
                        model1.eval()
                        model2.eval()

                        if args.dataset == "credit" or args.dataset == "pubmed":
                            val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features.cuda(), labels, Adj.cuda(),
                                                                             isPyG=False)
                        else:
                            val_loss, val_accu = self.get_loss_learnable_adj(model2, val_mask, features.cuda(), labels, Adj.cuda(),
                                                                             isPyG=True)

                        if val_accu > best_val_accu:
                            best_val_accu = val_accu
                            print("Val Loss {:.4f}, Val Accuracy {:.4f}".format(val_loss, val_accu))
                            if args.dataset == "credit" or args.dataset == "pubmed":
                                test_loss_, test_accu_ = self.get_loss_learnable_adj(model2, test_mask, features.cuda(),
                                                                                     labels, Adj.cuda(), isPyG=False)
                            else:
                                test_loss_, test_accu_ = self.get_loss_learnable_adj(model2, test_mask, features.cuda(),
                                                                                     labels, Adj.cuda(), isPyG=True)
                            print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss_, test_accu_))

            validation_accu.append(best_val_accu.item())
            model1.eval()
            model2.eval()

            with torch.no_grad():
                print("Test Loss {:.4f}, test Accuracy {:.4f}".format(test_loss_, test_accu_))
                test_accu.append(test_accu_.item())



            # Random idx
            # choose the target nodes
            # idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0] * args.nlabel)))
            # seed = args.seeds[trial]
            # np.random.seed(seed)
            # random.seed(seed)

            if args.save_testset == 1:
                # save testset
                idx_attack = np.array(random.sample(range(Adj.shape[0]), int(Adj.shape[0] * 0.1)))
                save_list(idx_attack, "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(trial) + "_.idx")
            else:
                # # load

                # run 100 times
                if args.run_all_testset == 1:
                    idx_attack_all = []
                    for k in range(args.ntrials):
                        idx_attack = []
                        idx_attack = read_list(idx_attack,
                                               "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(
                                                   k) + "_.idx")
                        idx_attack = np.array(idx_attack)
                        idx_attack_all.append(idx_attack)
                else: #run normal 10 times
                    idx_attack = []
                    idx_attack = read_list(idx_attack,
                                           "./Dataset/testset/" + args.dataset + "/idx_attack_" + args.dataset + "_trial_" + str(
                                               trial) + "_.idx")
                    idx_attack = np.array(idx_attack)

            # Do reconstruction metric
            if args.run_all_testset == 1:
                auroc, avg_prec = self.reconstruction_metric(original_adj, Adj.cpu().detach().numpy(), idx_attack_all,
                                                             args.dataset,
                                                             trial, args.save_testset, args.run_all_testset,
                                                             args.ntrials)
            else:
                auroc, avg_prec = self.reconstruction_metric(original_adj, Adj.cpu().detach().numpy(), idx_attack, args.dataset,
                                                             trial, args.save_testset, args.run_all_testset,
                                                             args.ntrials)

            # # Do reconstruction metric
            # auroc, avg_prec = self.reconstruction_metric(original_adj, Adj.cpu().detach().numpy(), idx_attack, args.dataset, trial, args.save_testset)
            avg_auroc.append(auroc)
            avg_avg_prec.append(avg_prec)
            print("trial", trial)

        print("args.ntrials", args.ntrials)
        print("avg_auroc", avg_auroc)
        print("reconstructed auroc mean", np.mean(avg_auroc))
        print("avg_avg_prec", avg_avg_prec)
        print("reconstructed avg_prec mean", np.mean(avg_avg_prec))

        print("reconstructed auroc std", np.std(avg_auroc))
        print("reconstructed avg_prec std", np.std(avg_avg_prec))

        self.print_results(validation_accu, test_accu)

    def print_results(self, validation_accu, test_accu):
        print(test_accu)
        print("std of test accuracy", np.std(test_accu))
        print("average of test accuracy", np.mean(test_accu))
        print(validation_accu)
        print("std of val accuracy", np.std(validation_accu))
        print("average of val accuracy", np.mean(validation_accu))


    def reconstruction_metric(self, ori_adj, inference_adj, idx, dataset="none", trial=000, save_testset=0, run_all_testset=0, num_test=0):
        auroc = 0
        avg_prec = 0

        if run_all_testset == 1:
            print("Testing all testset in each run")
            # return all test set in idx !
            # Note: you can't save test indexes when this is turned on. Reason: it's obvious!
            all_auc = []
            all_avg_prec = []
            for k in range(num_test):
                print("each_test", k)
                print("ori_adj b4", ori_adj[:10])  # [[0 0 0 ... 0 0 0]
                print("inference_adj", inference_adj[:10])  # [[5.7996154e-02 1.7944765e-05 1.3481597e-05 ... ]
                print("ori_adj.shape", ori_adj.shape)  # 2708x2708 --> It is converted to numpy and it is 0, 1 encoded!
                print("inference_adj.shape",
                      inference_adj.shape)  # 2708 x 2708 --> It is converted to numpy and it's probabilities
                # print("idx", idx) # They are nodes that u wanna attack e.g [ 271 1310  220  968  618  966 ...] it is 10% of the total number of nodes
                print("idx.shape", idx[k].shape)  # 270

                # get the real and predicted edges for the idx of interest! Then compute their
                real_edge = ori_adj[idx[k], :][:, idx[k]].reshape(-1)
                # print(type(real_edge)) #numpymatrix
                # For some reason, the real_edge has an extra dimension. Flatten!
                real_edge = (np.asarray(real_edge)).flatten()
                # print("real_edge after", real_edge) #[0. 0. 0. ...]
                # print("real_edge.shapessssss", real_edge1.shape)
                print("real_edge.shape", real_edge.shape)  # 72900
                # print("real_edge", list(real_edge)) # 72900
                # edge_0 = np.where(real_edge == 0)[0]
                # edge_1 = np.where(real_edge == 1)[0]
                # print("len(edge_0)", len(edge_0), "len(edge_1)", len(edge_1))

                pred_edge = inference_adj[idx[k], :][:, idx[k]].reshape(
                    -1)
                # print("pred_edge after", pred_edge) # [0.         0.63648593 0.5467699  ...]
                print("pred_edge.shape", pred_edge.shape)  # 72900
                # fpr, tpr, threshold = roc_curve(real_edge, pred_edge) #old AUROC on all

                # This should go for each!
                # load
                index_delete = []
                # index_delete_all = []
                index_delete = read_list(index_delete,
                                         "./Dataset/testset/" + dataset + "/index_delete_" + dataset + "_trial_" + str(
                                             k) + "_.idx")
                index_delete = np.array(index_delete)
                # index_delete_all.append(index_delete)

                print("int(len(real_edge)-2*np.sum(real_edge))", int(len(real_edge) - 2 * np.sum(real_edge)))  # 72652
                # print("index_delete", index_delete) # [42444 19960 68639 ...
                print("index_delete.shape", index_delete.shape)  # 72652 # still about 80% that u wanna delete!
                real_edge = np.delete(real_edge, index_delete)
                # print("real_edge real_edge", real_edge)
                pred_edge = np.delete(pred_edge, index_delete)
                # print("pred_edge pred_edge", pred_edge)
                # print("real_edge", real_edge[:10]) #[0. 0. 1. 1.
                # It is 72776 - 72652 = 124 x 2 = 248
                print("real_edge.shape", real_edge.shape)  # 248 nodes / integers!
                # print("pred_edge", pred_edge[:10]) #[0.74064773 0.4564297  0.906284   0.53555965
                print("pred_edge.shape", pred_edge.shape)  # 248 nodes / integers!
                # auroc = auc(fpr, tpr) #old AUROC on all

                # New: AUROC on balanced
                print("real_edge real_edge", list(real_edge))
                fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
                each_auroc = auc(fpr, tpr)

                each_avg_prec = average_precision_score(real_edge, pred_edge)
                print("Inference attack AUC: %f AP: %f" % (each_auroc, each_avg_prec))
                all_auc.append(each_auroc)
                all_avg_prec.append(each_avg_prec)

            auroc = all_auc
            avg_prec = all_avg_prec


        else:
            print("ori_adj b4", ori_adj[:10]) # [[0 0 0 ... 0 0 0]
            print("inference_adj", inference_adj[:10]) #[[5.7996154e-02 1.7944765e-05 1.3481597e-05 ... ]
            print("ori_adj.shape", ori_adj.shape)  # 2708x2708 --> It is converted to numpy and it is 0, 1 encoded!
            print("inference_adj.shape",
                  inference_adj.shape)  # 2708 x 2708 --> It is converted to numpy and it's probabilities
            # print("idx", idx) # They are nodes that u wanna attack e.g [ 271 1310  220  968  618  966 ...] it is 10% of the total number of nodes
            print("idx.shape", idx.shape)  # 270

            # get the real and predicted edges for the idx of interest! Then compute their
            real_edge = ori_adj[idx, :][:, idx].reshape(-1)
            # print(type(real_edge)) #numpymatrix
            # For some reason, the real_edge has an extra dimension. Flatten!
            real_edge = (np.asarray(real_edge)).flatten()
            # print("real_edge after", real_edge) #[0. 0. 0. ...]
            # print("real_edge.shapessssss", real_edge1.shape)
            print("real_edge.shape", real_edge.shape) # 72900
            # print("real_edge", list(real_edge)) # 72900
            # edge_0 = np.where(real_edge == 0)[0]
            # edge_1 = np.where(real_edge == 1)[0]
            # print("len(edge_0)", len(edge_0), "len(edge_1)", len(edge_1))

            pred_edge = inference_adj[idx, :][:, idx].reshape(
                -1)
            # print("pred_edge after", pred_edge) # [0.         0.63648593 0.5467699  ...]
            print("pred_edge.shape", pred_edge.shape) # 72900
            # fpr, tpr, threshold = roc_curve(real_edge, pred_edge) #old AUROC on all

            if save_testset == 1: #save the test set for evaluation
                # save test index
                index = np.where(real_edge == 0)[0]
                # print("index", index) #[    0     1     2 ... ]
                print("index.shape", index.shape)  # (72776,). This is like 80%
                index_delete = np.random.choice(index, size=int(len(real_edge) - 2 * np.sum(real_edge)), replace=False)
                save_list(index_delete, "./Dataset/testset/" + dataset + "/index_delete_" + dataset + "_trial_" + str(trial) + "_.idx")
            else:
                # load
                index_delete = []
                index_delete = read_list(index_delete, "./Dataset/testset/" + dataset + "/index_delete_" + dataset + "_trial_" +str(trial)+"_.idx")
                index_delete = np.array(index_delete)

            print("int(len(real_edge)-2*np.sum(real_edge))", int(len(real_edge) - 2 * np.sum(real_edge)))  # 72652
            # print("index_delete", index_delete) # [42444 19960 68639 ...
            print("index_delete.shape", index_delete.shape)  # 72652 # still about 80% that u wanna delete!
            real_edge = np.delete(real_edge, index_delete)
            # print("real_edge real_edge", real_edge)
            pred_edge = np.delete(pred_edge, index_delete)
            # print("pred_edge pred_edge", pred_edge)
            # print("real_edge", real_edge[:10]) #[0. 0. 1. 1.
            # It is 72776 - 72652 = 124 x 2 = 248
            print("real_edge.shape", real_edge.shape)  # 248 nodes / integers!
            # print("pred_edge", pred_edge[:10]) #[0.74064773 0.4564297  0.906284   0.53555965
            print("pred_edge.shape", pred_edge.shape)  # 248 nodes / integers!
            # auroc = auc(fpr, tpr) #old AUROC on all

            # New: AUROC on balanced
            print("real_edge real_edge", list(real_edge))
            fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
            auroc = auc(fpr, tpr)

            avg_prec = average_precision_score(real_edge, pred_edge)
            print("Inference attack AUC: %f AP: %f" % (auroc, avg_prec))
        return auroc, avg_prec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('-epochs_adj', type=int, default=2000, help='Number of epochs to learn the adjacency.')
    parser.add_argument('-lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('-lr_adj', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-w_decay_adj', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('-hidden_adj', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('-dropout1', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout2', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj1', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dropout_adj2', type=float, default=0.25, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-dataset', type=str, default='cora', help='See choices',
                        choices=['cora', 'cora_ml', 'bitcoin', 'chameleon', 'credit', 'citeseer', 'pubmed'])
    parser.add_argument('-nlayers', type=int, default=2, help='#layers')
    parser.add_argument('-nlayers_adj', type=int, default=2, help='#layers')
    parser.add_argument('-patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-devices', help='Get devices auto assigned by condor')
    parser.add_argument('-ntrials', type=int, default=1, help='Number of trials')
    parser.add_argument('-seeds', nargs='+', default=[1050154401, 87952126, 461858464, 2251922041, 2203565404,
                                                      2569991973, 569824674, 2721098863, 836273002, 2935227127])
    parser.add_argument('-k', type=int, default=20, help='k for initializing with knn')
    parser.add_argument('-ratio', type=int, default=20, help='ratio of ones to select for each mask')
    parser.add_argument('-epoch_d', type=float, default=5,
                        help='epochs_adj / epoch_d of the epochs will be used for training only with DAE.')
    parser.add_argument('-lambda_', type=float, default=0.1, help='regularizing the loss')
    parser.add_argument('-nr', type=int, default=5, help='ratio of zeros to ones')
    parser.add_argument('-knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
    parser.add_argument('-model', type=str, default="exp_intersection", help='See choices',
                        choices=['end2end', 'normal', 'pairwise_sim', 'fidelity', 'exp_intersection']) #default="end2end". Normal will give the default performance!
    parser.add_argument('-i', type=int, default=6)
    parser.add_argument('-non_linearity', type=str, default='elu')
    parser.add_argument('-normalization', type=str, default='sym')
    parser.add_argument('-gen_mode', type=int, default=0) #FP
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-noise', type=str, default="mask", choices=['mask', 'normal'])
    parser.add_argument('-loss', type=str, default="mse", choices=['mse', 'bce'])
    parser.add_argument('-attack_type', type=str, default='gsef_concat', 
                        choices=['gsef_concat', 'gsef_mult', 'gsef', 'gse', 'explainsim', 'featuresim', 'slaps'])
    parser.add_argument('-explanation_method', type=str, default='grad',
                        choices=['grad', 'gradinput', 'zorro-soft', 'zorro-hard', 'graphlime', 'gnn-explainer'])
    parser.add_argument('-load_exp_model', type=int, default=0, choices=[1, 0],
                        help='1 = explanation model will be loaded. If 0, no need for loading explanation model')
    parser.add_argument('-get_predicted_labels', type=int, default=0, choices=[1, 0],
                        help='1 = use the released trained model to retrieve labels. If 0, use normal groundtruth')
    parser.add_argument('-use_exp_as_reconstruction_loss', type= int, default=0, choices=[1,0], 
                        help='1 = explanation will be used as loss. if 0, explanation will not be used with loss function')
    parser.add_argument('-get_fidelity', type=int, default=0, choices=[1, 0],
                        help='1 = run fidelity. if 0, no fidelity will be ran')
    parser.add_argument('-use_subgraph', type=int, default=0, choices=[1,0], help='run the subgraph experiment')
    parser.add_argument('-get_intersection', type=int, default=0, choices=[1, 0],
                        help='1 = run intersection and sparsity. if 0, no intersection will be ran')
    parser.add_argument('-save_testset', type=int, default=0, choices=[1, 0],
                        help='1 = Save fixed testset. if 0, no testset is saved')
    parser.add_argument('-run_all_testset', type=int, default=0, choices=[1,0], help='1= for each experiment, run all 10 test! Total will 10 runs * 10 testset=100, 0= normal 10 times run')
    parser.add_argument('-use_defense', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help='if 0, no defense, 1 = use the defense that splits into multiple explanations and perturb and add together again. 2 = Do multi-bit piecewise mechanism i.e no explanation splitting, 3 = Gaussian, 4 = Multibit,  5 = Randomized response')
    parser.add_argument('-epsilon', type=float, default=0.0001, help='epsilon for perturbing the explanations')
    parser.add_argument('-num_exp_in_each_split', type=int, default=10, help='Number of explanation vector in each split for defense 1. Input any number between 2 and num_feature-1')


    args = parser.parse_args()

    print("model_type:", args.model)
    print("dataset", args.dataset)

    if args.use_exp_as_reconstruction_loss == 1:
        print("=================Using the explanation in the loss function====================================")

    experiment = Experiment()
    start_time = time.time()

    if args.model == "end2end":
        experiment.train_end_to_end(args)
    elif args.model == "normal":
        experiment.train_test_normal(args)
    elif args.model == "pairwise_sim":
        experiment.pairwise_similarity(args)
    elif args.model == "fidelity":
        experiment.run_fidelity(args)
    elif args.model == "exp_intersection":
        experiment.run_intersection(args)

    end_time = time.time()

    total_time = end_time - start_time
    print("total time", total_time / args.ntrials)
