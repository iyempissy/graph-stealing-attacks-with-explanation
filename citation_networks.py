# The MIT License

# Copyright (c) 2016 Thomas Kipf

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pickle as pkl
import sys
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import accuracy
from torch_geometric.utils import dense_to_sparse, convert, from_networkx
from defenses import split_explanation

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 7)  # instead of 7, it should be the size of the labels

warnings.simplefilter("ignore")


def get_pretrained_labels(path, model, features, edges, gt_labels):
    print("======================= Using prediction from target / released model as GT =======================")
    # Note that the gt_labels is only used for computing accuracy

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(path))
    model.eval()

    print("features", features)
    print("features.shape", features.shape)
    print("edges", edges.shape)
    logits = model(features, edges)
    logp = logits  # F.log_softmax(logits, 1)
    accu = accuracy(logp, gt_labels)

    pred_labels = torch.max(logp, 1)[1]

    print("pred_labels================", pred_labels)
    print("Accuracy is ===>", accu)

    return pred_labels


def plot_explanations(explanation, type, data_name, labels):
    print("About to plot!")
    feat = TSNE(n_components=2, random_state=10).fit_transform(explanation[1].reshape(-1, 1))
    g = sns.scatterplot(feat[:, 0], feat[:, 1], legend='full', palette=palette, s=150)  # hue = labels,

    # g.xaxis.set_tick_params(labelsize=50)
    # g.yaxis.set_tick_params(labelsize=50)
    g.set(xticklabels=[], yticklabels=[])
    g.set(xlabel=None, ylabel=None)

    plt.savefig("./1" + data_name + "_" + type + "expplot.pdf", format='pdf',
                dpi=1200, bbox_inches="tight")

    plt.close()
    plt.show()


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# for zorro cora and coraml
def load_soft_mask(path_prefix, node):
    path = path_prefix + "_node_" + str(node) + ".npz"
    save = np.load(path)
    node_mask = save["node_mask"]
    feature_mask = save["feature_mask"]
    execution_time = save["execution_time"]
    if execution_time is np.inf:
        return node_mask, feature_mask
    else:
        return node_mask, feature_mask, float(execution_time)


# Zorro explanation #Hardmasking
def load_minimal_nodes_and_features_sets_zorro(path_prefix, node, check_for_initial_improves=False, isBitcoin=False):
    if isBitcoin:
        path = path_prefix + str(node)
    else:
        path = path_prefix + "_node_" + str(node) + ".npz"

    save = np.load(path, allow_pickle=True)

    saved_node = save["node"]
    if saved_node != node:
        raise ValueError("Other node then specified", saved_node, node)
    number_of_sets = save["number_of_sets"]

    minimal_nodes_and_features_sets = []

    if number_of_sets > 0:

        features_label = "features_"
        nodes_label = "nodes_"
        selection_label = "selection_"

        for i in range(number_of_sets):
            selected_nodes = save[nodes_label + str(i)]
            selected_features = save[features_label + str(i)]
            executed_selections = None  # save[selection_label + str(i)]

            minimal_nodes_and_features_sets.append((selected_nodes, selected_features, executed_selections))

    if check_for_initial_improves:
        try:
            initial_node_improve = save["initial_node_improve"]
        except KeyError:
            initial_node_improve = None

        try:
            initial_feature_improve = save["initial_feature_improve"]
        except KeyError:
            initial_feature_improve = None

        return minimal_nodes_and_features_sets, initial_node_improve, initial_feature_improve
    else:
        return minimal_nodes_and_features_sets


def load_citation_network(dataset_str, use_exp=False, concat_feat_with_exp=False, exp_only_as_feature=False,
                          exp_type="grad", use_exp_with_loss=0, get_fidelity=0, use_defense=0, get_intersection=0,
                          epsilon=0, num_exp_in_each_split=10, get_predicted_labels=0, path=None, released_model=None):
    # if use_exp=True and concat_feat_with_exp=True, then do concatenation.
    # if use_exp=True and concat_feat_with_exp=False, then it does element wise multiplication.

    if use_exp:
        print("exp_type======", exp_type)
    else:
        print("No explanation is used!")

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data_tf/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data_tf/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    # print("idx_train", idx_train) #range(0, 140)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    features = torch.FloatTensor(features.todense())
    original_features = features
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    explanations = None
    original_exp = None
    perturbed_exp = None

    # use explanations
    if use_exp:
        if exp_type == "zorro-soft":
            exp_folder = "Cora_Explanations/Zorro_soft_Cora/gcn_2_layers_explanation"
            print("xxxxxxxxxxxx This is zorro-soft xxxxxxxxxxxx")
        elif exp_type == "zorro-hard":
            exp_folder = "Cora_Explanations/Zorro_hard_Cora/gcn_2_layers_explanation"
            print("xxxxxxxxxxxx This is zorro-hard xxxxxxxxxxxx")
        elif exp_type == "grad":
            exp_folder = "Cora_Explanations/Grad_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is grad xxxxxxxxxxxx")
        elif exp_type == "grad-untrained":
            exp_folder = "Cora_Explanations/Grad_untrained_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is grad untrained xxxxxxxxxxxx")
        elif exp_type == "gnn-explainer":
            exp_folder = "Cora_Explanations/GNNExplainer_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is GNNExplainer xxxxxxxxxxxx")
        elif exp_type == "graphlime":
            exp_folder = "Cora_Explanations/GraphLime_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime xxxxxxxxxxxx")
        elif exp_type == "graphlime01":  # graphlime with rho of 0.1
            exp_folder = "Cora_Explanations/GraphLime_Cora_0.1/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime 0.1 xxxxxxxxxxxx")
        elif exp_type == "gradinput-untrained":
            exp_folder = "Cora_Explanations/GradInput_untrained_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput untrained xxxxxxxxxxxx")
        else:  # for gradinput
            exp_folder = "Cora_Explanations/GradInput_Cora/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput xxxxxxxxxxxx")

        all_feat_exp = []
        for i in range(0, len(features)):
            if exp_type == "zorro-soft":
                _, feat_exp_i, _ = load_soft_mask(exp_folder, i)
                # # remove extra dimension
                feat_exp_i = (np.asarray(feat_exp_i)).flatten()
            elif exp_type == "zorro-hard":
                feat_exp_i = \
                load_minimal_nodes_and_features_sets_zorro(exp_folder, i, check_for_initial_improves=False)[0][1]
                # remove extra dimension
                feat_exp_i = (np.asarray(feat_exp_i)).flatten()
            else:
                feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
            all_feat_exp.append(feat_exp_i)

        # convert list of arrays to single array!
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        if exp_type == "gnn-explainer":  # remove extra dimension
            all_feat_exp = np.squeeze(all_feat_exp)

        # print(all_feat_exp.shape) #(2708, 1433)
        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # plot_explanations(exp_features, exp_type, dataset_str, labels)

        # Defense. Change the explanation vector here!
        if use_defense != 0:
            original_exp = exp_features  # make a copy of this
            exp_features = split_explanation(exp_features, num_exp_in_each_split, eps=epsilon, defense_type=use_defense)
            perturbed_exp = exp_features
        # elif use_defense == 2: #multi piecewise only
        #     exp_features = split_explanation(exp_features, 0, defense_type=use_defense)

        if use_exp_with_loss == 1:
            features = features
            explanations = exp_features
            print("using explanations with the loss function")
        elif get_fidelity == 1:
            features = features
            explanations = exp_features
            print("Run fidelity: explanation = explanation, features = features")
        elif exp_only_as_feature:
            features = exp_features  # i.e using only explanations
            print("explanation now features", features)
            print("explanation now features", features.shape)
            print("********************** Explanation only **********************")
        else:
            # concat features
            if concat_feat_with_exp:
                final_feature = torch.cat((features, exp_features), 1)
                print("********************** Concat feat and exp **********************")
            else:
                # Do element wise multiplication of features and explanations!
                final_feature = torch.mul(features, exp_features)
                print("********************** Elem feat and exp **********************")
            # print(final_feature)
            # print(final_feature.shape)
            features = final_feature

    nfeats = features.shape[1]

    # print("graph", graph[1]) #should be a dict!
    # convert to networkx adjacency!
    G = nx.Graph(graph)

    # print(nx.info(G))
    # print(nclasses)

    original_adj = nx.adjacency_matrix(G).todense()

    data = from_networkx(G)
    data.x = original_features
    print("data", data)
    print("edge_index", data.edge_index)
    print("edge_index shape", data.edge_index.shape)

    # Assumption that the attacker can retrive labels from the released model. We used the default features for extracting the labels
    if get_predicted_labels == 1:
        labels = get_pretrained_labels(path, released_model, data.x, data.edge_index, labels)

    # print("original_adj", original_adj)
    # print(original_adj.shape) #2708 x2708
    if use_exp_with_loss == 1:
        return explanations, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, path

    elif get_intersection == 1:
        # Note that to run this, use_defense has to be set
        return original_exp, perturbed_exp

    elif get_fidelity == 1:  # run_fidelity
        return explanations, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, data.edge_index, path
    else:
        # default
        return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj, path
