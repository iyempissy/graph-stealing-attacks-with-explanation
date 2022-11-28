import warnings
import torch
import sys
from collections import namedtuple

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

from citation_networks import load_citation_network, sample_mask, load_minimal_nodes_and_features_sets_zorro, \
    load_soft_mask, plot_explanations, get_pretrained_labels

from torch_geometric.utils import convert, from_networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from read_bitcoin import *
from model import GCN_PyG
from defenses import split_explanation
from read_chameleon import *
from read_credit import *


warnings.simplefilter("ignore")


def load_ogb_data(dataset_str, use_exp=False, concat_feat_with_exp=False):
    # if use_exp=True and concat_feat_with_exp=True, then do concatenation.
    # if use_exp=True and concat_feat_with_exp=False, then it does element wise multiplication.
    
    
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(dataset_str)

    data = dataset[0]
    features = data.x
    # nfeats = data.num_features
    nclasses = dataset.num_classes
    labels = data.y


    split_idx = dataset.get_idx_split()

    train_mask = sample_mask(split_idx['train'], data.x.shape[0])
    val_mask = sample_mask(split_idx['valid'], data.x.shape[0])
    test_mask = sample_mask(split_idx['test'], data.x.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    # use explanations
    if use_exp:
        all_feat_exp = []
        for i in range(0, len(features)):
            feat_exp_i = torch.load("Doesnotexist-Ogb_Explanations/Grad_Ogb/feature_masks_node=" + str(i))
            all_feat_exp.append(feat_exp_i)

        # convert list of arrays to single array!
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        # print(all_feat_exp.shape) #(2708, 1433)
        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # concat features
        if concat_feat_with_exp:
            final_feature = torch.cat((features, exp_features), 1)
        else:
            # Do element wise multiplication of features and explanations!
            final_feature = torch.mul(features, exp_features)
            
        # print(final_feature)
        # print(final_feature.shape)
        features = final_feature



    nfeats = features.shape[1]
    G = convert.to_networkx(data, to_undirected=True)
    original_adj = nx.adjacency_matrix(G).todense()

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj





def load_dataset_text(file_name):
    """Load a graph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    # np_load_old = np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def create_train_val_test_mask(data, num_train_per_class=20, num_classes=None, num_val=500, num_test=1000, ):
    import numpy as np
    # fix seed for selecting train_mask
    # rng = np.random.default_rng(seed=42 * 20200909)
    rng = np.random.RandomState(seed=42 * 20200909)

    if num_classes is None:
        num_classes = torch.max(data.y)

    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero().view(-1)
        idx = idx[rng.permutation(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero().view(-1)
    remaining = remaining[rng.permutation(remaining.size(0))]

    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[remaining[:num_val]] = True

    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask


# # for zorro coraml
# def load_soft_mask(path_prefix, node):
#     path = path_prefix + "_node_" + str(node) + ".npz"
#     save = np.load(path)
#     node_mask = save["node_mask"]
#     feature_mask = save["feature_mask"]
#     execution_time = save["execution_time"]
#     if execution_time is np.inf:
#         return node_mask, feature_mask
#     else:
#         return node_mask, feature_mask, float(execution_time)



# for cora ml dataset
def load_cora_ml(dataset, use_exp=False, concat_feat_with_exp=False, exp_only_as_feature=False, exp_type="grad",
                 use_exp_with_loss=0, get_fidelity=0, use_defense=0, get_intersection=0, epsilon=0,
                 num_exp_in_each_split=10, get_predicted_labels=0, path=None, released_model=None):
    data_name = "cora_ml"
    
    if use_exp:
        print("exp_type======", exp_type)
    else:
        print("No explanation is used!")

    graph = load_dataset_text(dataset)
    A = graph['A']
    y = torch.tensor(graph['z'])

    x = np.load('./Dataset/w2v_embeddings.npy',allow_pickle=True)
    x = torch.tensor(x,dtype=torch.float)
    Acoo = A.tocoo()

    Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                  torch.LongTensor(Acoo.data.astype(np.int32)))

    edge_index = Apt._indices()

    print("edge_index.shape", edge_index.shape)

    data = Data(x=x,edge_index=edge_index,y=y)
    # print(data.y)
    print("data.x features shape", data.x.shape)
    print("data.x features", data.x)
    print("edge_index", data.edge_index)
    # print("edge_index", data.edge_index.shape)
    data.train_mask, data.val_mask, data.test_mask = create_train_val_test_mask(data)
    Dataset = namedtuple("Dataset", "num_node_features num_classes")
    dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)
    results_path = "cora_ml"

    features = data.x
    explanations = None
    perturbed_exp = None
    original_exp = None

    # use explanations
    if use_exp:
        if exp_type == "zorro-soft":
            exp_folder = "Cora_ml_Explanations/Zorro_soft_Cora_ml/gcn_2_layers_explanation"
            print("xxxxxxxxxxxx This is zorro-soft xxxxxxxxxxxx")
        elif exp_type == "zorro-hard":
            exp_folder = "Cora_ml_Explanations/Zorro_hard_Cora_ml/gcn_2_layers_explanation_t_3_r_1"
            print("xxxxxxxxxxxx This is zorro-hard xxxxxxxxxxxx")
        elif exp_type == "grad":
            exp_folder = "Cora_ml_Explanations/Grad_Cora_ml/feature_masks_node="
            print("xxxxxxxxxxxx This is grad xxxxxxxxxxxx")
        elif exp_type == "grad-untrained":
            exp_folder = "Cora_ml_Explanations/Grad_untrained_Cora_ml/feature_masks_node="
            print("xxxxxxxxxxxx This is grad untrained xxxxxxxxxxxx")
        elif exp_type == "gnn-explainer":
            exp_folder = "Cora_ml_Explanations/GNNExplainer_Cora_ml/feature_masks_node="
            print("xxxxxxxxxxxx This is GNNExplainer xxxxxxxxxxxx")
        elif exp_type == "graphlime":
            exp_folder = "Cora_ml_Explanations/GraphLime_Cora_ml_0.1/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime xxxxxxxxxxxx")
        # elif exp_type == "graphlime01": #graphlime with rho of 0.1
        #     exp_folder = "Cora_ml_Explanations/GraphLime_Cora_ml_0.1/feature_masks_node="
        #     print("xxxxxxxxxxxx This is GraphLime 0.1xxxxxxxxxxxx")
        elif exp_type == "gradinput-untrained":
            exp_folder = "Cora_ml_Explanations/GradInput_untrained_Cora_ml/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput untrained xxxxxxxxxxxx")
        else: # for gradinput
            exp_folder = "Cora_ml_Explanations/GradInput_Cora_ml/feature_masks_node="
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
                feat_exp_i = torch.load(exp_folder + str(i)) #load explanations
            all_feat_exp.append(feat_exp_i)

        # convert list of arrays to single array!
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        if exp_type == "gnn-explainer": #remove extra dimension
            all_feat_exp = np.squeeze(all_feat_exp)
            
        # print(all_feat_exp.shape) #(2708, 1433)
        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # plot_explanations(exp_features, exp_type, data_name, data.y)

        # Defense. Change the explanation vector here!
        if use_defense != 0:
            original_exp = exp_features  # make a copy of this
            exp_features = split_explanation(exp_features, num_exp_in_each_split, eps=epsilon, defense_type=use_defense)
            perturbed_exp = exp_features
        # elif use_defense == 2:  # multi piecewise only
        #     exp_features = split_explanation(exp_features, 0, defense_type=use_defense)


        if use_exp_with_loss == 1:
            features = features
            explanations = exp_features
            print("********************** using explanations with the loss function ********************** ")
        elif get_fidelity == 1:
            features = features
            explanations = exp_features
            print("Run fidelity: explanation = explanation, features = features")
        elif exp_only_as_feature:
            features = exp_features #i.e using only explanations
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




    G = convert.to_networkx(data, to_undirected=True)

    # print(nx.info(G))
    # print(max(data.y.numpy()) + 1)
    original_adj = nx.adjacency_matrix(G).todense()

    # Assumption that the attacker can retrive labels from the released model. 
    # We used the default features for extracting the labels
    if get_predicted_labels == 1:
        data.y = get_pretrained_labels(path, released_model, data.x, data.edge_index, data.y)

    # return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj
    if use_exp_with_loss == 1:
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path
    
    elif get_intersection == 1:
        # Note that to run this, use_defense has to be set
        return original_exp, perturbed_exp
    
    elif get_fidelity == 1: #run_fidelity
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, data.edge_index, path
    else:
        # default
        return features, features.shape[1], data.y, max(data.y.numpy()) + 1, data.train_mask, data.val_mask, \
               data.test_mask, original_adj, path







def load_bitcoin(dataset, use_exp=False, concat_feat_with_exp=False, exp_only_as_feature=False, exp_type="grad", 
                 use_exp_with_loss = 0, get_fidelity = 0, use_defense = 0, get_intersection = 0, epsilon=0, 
                 num_exp_in_each_split=10, get_predicted_labels=0, path = None, released_model = None):
    data_name = "bitcoin"
    g, labels, name = read_bitcoinalpha(dataset)
    A = nx.adjacency_matrix(g).todense()

    # print(nx.info(g))

    data = from_networkx(g)
    data.x = data.x.to(torch.float32)
    data.edge_attr = data.RATING

    data.y = np.array(labels)
    data.y = torch.from_numpy(data.y)
    num_nodes = A.shape[0]
    train_ratio = 0.8
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)

    train_mask = np.full_like(data.y, False, dtype=bool)
    train_mask[idx[:num_train]] = True
    test_mask = np.full_like(data.y, False, dtype=bool)
    test_mask[idx[num_train:]] = True

    data.train_mask, data.test_mask = torch.tensor(train_mask), torch.tensor(test_mask)

    print("data.train_mask", data.train_mask)

    # Used test_mask = val_mask!
    data.val_mask = torch.tensor(test_mask)


    Dataset = namedtuple("Dataset", "num_node_features num_classes")
    dataset = Dataset(data.x.shape[1], max(data.y).item() + 1)

    results_path = "Bitcoin_alpha"


    features = data.x
    explanations = None
    perturbed_exp = None
    original_exp = None

    # use explanations
    if use_exp:
        if exp_type == "zorro-soft":
            exp_folder = "Bitcoin_Explanations/Zorro_soft_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is zorro-soft xxxxxxxxxxxx")
        elif exp_type == "zorro-hard":
            exp_folder = "Bitcoin_Explanations/Zorro_hard_Bitcoin/feature_masks_node=" #gcn_2_layers_explanation_t_3_r_1
            print("xxxxxxxxxxxx This is zorro-hard xxxxxxxxxxxx")
        elif exp_type == "grad":
            exp_folder = "Bitcoin_Explanations/Grad_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is grad xxxxxxxxxxxx")
        elif exp_type == "grad-untrained":
            exp_folder = "Bitcoin_Explanations/Grad_untrained_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is grad untrained xxxxxxxxxxxx")
        elif exp_type == "gnn-explainer":
            exp_folder = "Bitcoin_Explanations/GNNExplainer_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is GNNExplainer xxxxxxxxxxxx")
        elif exp_type == "graphlime":
            exp_folder = "Bitcoin_Explanations/GraphLime_Bitcoin_0.1/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime xxxxxxxxxxxx")
        # elif exp_type == "graphlime01":  # graphlime with rho of 0.1
        #     exp_folder = "Bitcoin_Explanations/GraphLime_Bitcoin_0.1/feature_masks_node="
        #     print("xxxxxxxxxxxx This is GraphLime 0.1xxxxxxxxxxxx")
        elif exp_type == "gradinput-untrained":
            exp_folder = "Bitcoin_Explanations/GradInput_untrained_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput untrained xxxxxxxxxxxx")
        else:  # for gradinput
            exp_folder = "Bitcoin_Explanations/GradInput_Bitcoin/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput xxxxxxxxxxxx")

        all_feat_exp = []
        for i in range(0, len(features)):
            if exp_type == "zorro-soft":
                # _, feat_exp_i, _ = load_soft_mask(exp_folder, i)
                # # # remove extra dimension
                # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()
            elif exp_type == "zorro-hard":
                # feat_exp_i = load_minimal_nodes_and_features_sets_zorro(exp_folder, i, 
                # check_for_initial_improves=False, isBitcoin=True)[0][1]
                # # remove extra dimension
                # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()
            else:
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()

            all_feat_exp.append(feat_exp_i)
        # print("all_feat_exp", all_feat_exp)

        # convert list of arrays to single array!
        # if exp_type == "zorro-hard":
        #     all_feat_exp = np.stack(all_feat_exp, axis=0)
        #     all_feat_exp = all_feat_exp.cpu()
        # else:
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        # if exp_type == "gnn-explainer" or exp_type == "grad":  # remove extra dimension
        all_feat_exp = np.squeeze(all_feat_exp) #seems like all of the explanations have extra dim

        # print("features", features) #floating number
        print("features.shape", features.shape) #(3783, 8)

        print(all_feat_exp.shape) #(3783, 8)

        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # plot_explanations(exp_features, exp_type, data_name, data.y)

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
            print("********************** using explanations with the loss function ********************** ")
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

    G = convert.to_networkx(data, to_undirected=True)
    # print(nx.info(G))
    # print("num classes". max(data.y.numpy()) + 1)
    original_adj = nx.adjacency_matrix(G).todense()

    # Assumption that the attacker can retrive labels from the released model. 
    # We used the default features for extracting the labels
    if get_predicted_labels == 1:
        data.y = get_pretrained_labels(path, released_model, data.x, data.edge_index, data.y)

    # return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj
    if use_exp_with_loss == 1:
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path

    elif get_intersection == 1:
        # Note that to run this, use_defense has to be set
        return original_exp, perturbed_exp

    elif get_fidelity == 1: #get_fidelity
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, data.edge_index, path
    else:
        # default
        return features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path











def load_chameleon(dataset, use_exp=False, concat_feat_with_exp=False, exp_only_as_feature=False, exp_type="grad",
                 use_exp_with_loss = 0, get_fidelity = 0, use_defense = 0, get_intersection = 0, epsilon=0,
                 num_exp_in_each_split=10, get_predicted_labels=0, path = None, released_model = None):
    data_name = "chameleon"
    data = read_chameleon_dataset(dataset)



    features = data.x
    explanations = None
    perturbed_exp = None
    original_exp = None

    # use explanations
    if use_exp:
        if exp_type == "zorro-soft":
            exp_folder = "Chameleon_Explanations/Zorro_soft_Chameleon/feature_masks_node"
            print("xxxxxxxxxxxx This is zorro-soft xxxxxxxxxxxx")
        elif exp_type == "zorro-hard":
            exp_folder = "Chameleon_Explanations/Zorro_hard_Chameleon/feature_masks_node=" #gcn_2_layers_explanation_t_3_r_1
            print("xxxxxxxxxxxx This is zorro-hard xxxxxxxxxxxx")
        elif exp_type == "grad":
            exp_folder = "Chameleon_Explanations/Grad_Chameleon/feature_masks_node="
            print("xxxxxxxxxxxx This is grad xxxxxxxxxxxx")
        elif exp_type == "grad-untrained":
            exp_folder = "Chameleon_Explanations/Grad_untrained_Chameleon/feature_masks_node="
            print("xxxxxxxxxxxx This is grad untrained xxxxxxxxxxxx")
        elif exp_type == "gnn-explainer":
            exp_folder = "Chameleon_Explanations/GNNExplainer_Chameleon/feature_masks_node="
            print("xxxxxxxxxxxx This is GNNExplainer xxxxxxxxxxxx")
        elif exp_type == "graphlime":
            exp_folder = "Chameleon_Explanations/GraphLime_Chameleon_0.1/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime xxxxxxxxxxxx")
        # elif exp_type == "graphlime01":  # graphlime with rho of 0.1
        #     exp_folder = "Chameleon_Explanations/GraphLime_Chameleon_0.1/feature_masks_node="
        #     print("xxxxxxxxxxxx This is GraphLime 0.1xxxxxxxxxxxx")
        elif exp_type == "gradinput-untrained":
            exp_folder = "Chameleon_Explanations/GradInput_untrained_Chameleon/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput untrained xxxxxxxxxxxx")
        else:  # for gradinput
            exp_folder = "Chameleon_Explanations/GradInput_Chameleon/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput xxxxxxxxxxxx")

        all_feat_exp = []
        for i in range(0, len(features)):
            if exp_type == "zorro-soft":
                # _, feat_exp_i, _ = load_soft_mask(exp_folder, i)
                # # # remove extra dimension
                # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()
            elif exp_type == "zorro-hard":
                # feat_exp_i = load_minimal_nodes_and_features_sets_zorro(exp_folder, i,
                # check_for_initial_improves=False, isChameleon=True)[0][1]
                # # remove extra dimension
                # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()
            else:
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()

            all_feat_exp.append(feat_exp_i)
        # print("all_feat_exp", all_feat_exp)

        # convert list of arrays to single array!
        # if exp_type == "zorro-hard":
        #     all_feat_exp = np.stack(all_feat_exp, axis=0)
        #     all_feat_exp = all_feat_exp.cpu()
        # else:
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        # if exp_type == "gnn-explainer" or exp_type == "grad":  # remove extra dimension
        all_feat_exp = np.squeeze(all_feat_exp) #seems like all of the explanations have extra dim

        # print("features", features) #floating number
        print("features.shape", features.shape) #(3783, 8)

        print(all_feat_exp.shape) #(3783, 8)

        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # plot_explanations(exp_features, exp_type, data_name, data.y)

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
            print("********************** using explanations with the loss function ********************** ")
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

    G = convert.to_networkx(data, to_undirected=True)
    # print(nx.info(G))
    # print("num classes". max(data.y.numpy()) + 1)
    original_adj = nx.adjacency_matrix(G).todense()

    # Assumption that the attacker can retrive labels from the released model.
    # We used the default features for extracting the labels
    if get_predicted_labels == 1:
        data.y = get_pretrained_labels(path, released_model, data.x, data.edge_index, data.y)

    # return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj
    if use_exp_with_loss == 1:
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path

    elif get_intersection == 1:
        # Note that to run this, use_defense has to be set
        return original_exp, perturbed_exp

    elif get_fidelity == 1: #get_fidelity
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, data.edge_index, path
    else:
        # default
        return features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path














def load_credit(dataset, use_exp=False, concat_feat_with_exp=False, exp_only_as_feature=False, exp_type="grad",
                 use_exp_with_loss = 0, get_fidelity = 0, use_defense = 0, get_intersection = 0, epsilon=0,
                 num_exp_in_each_split=10, get_predicted_labels=0, path = None, released_model = None):
    data_name = "credit"
    data = read_credit_dataset(dataset, label_number=15000)



    features = data.x
    explanations = None
    perturbed_exp = None
    original_exp = None

    # use explanations
    if use_exp:
        if exp_type == "zorro-soft":
            # exp_folder = "Credit_Explanations/Zorro_soft_Credit/feature_masks_node"
            exp_folder = "Credit_Explanations/Zorro_soft_Credit/gcn_2_layers_explanation"
            print("xxxxxxxxxxxx This is zorro-soft xxxxxxxxxxxx")
        elif exp_type == "zorro-hard":
            exp_folder = "Credit_Explanations/Zorro_hard_Credit/feature_masks_node=" #gcn_2_layers_explanation_t_3_r_1
            print("xxxxxxxxxxxx This is zorro-hard xxxxxxxxxxxx")
        elif exp_type == "grad":
            exp_folder = "Credit_Explanations/Grad_Credit/feature_masks_node="
            print("xxxxxxxxxxxx This is grad xxxxxxxxxxxx")
        elif exp_type == "grad-untrained":
            exp_folder = "Credit_Explanations/Grad_untrained_Credit/feature_masks_node="
            print("xxxxxxxxxxxx This is grad untrained xxxxxxxxxxxx")
        elif exp_type == "gnn-explainer":
            exp_folder = "Credit_Explanations/GNNExplainer_Credit/feature_masks_node="
            print("xxxxxxxxxxxx This is GNNExplainer xxxxxxxxxxxx")
        elif exp_type == "graphlime":
            exp_folder = "Credit_Explanations/GraphLime_Credit_0.1/feature_masks_node="
            print("xxxxxxxxxxxx This is GraphLime xxxxxxxxxxxx")
        # elif exp_type == "graphlime01":  # graphlime with rho of 0.1
        #     exp_folder = "Credit_Explanations/GraphLime_Credit_0.1/feature_masks_node="
        #     print("xxxxxxxxxxxx This is GraphLime 0.1xxxxxxxxxxxx")
        elif exp_type == "gradinput-untrained":
            exp_folder = "Credit_Explanations/GradInput_untrained_Credit/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput untrained xxxxxxxxxxxx")
        else:  # for gradinput
            exp_folder = "Credit_Explanations/GradInput_Credit/feature_masks_node="
            print("xxxxxxxxxxxx This is gradinput xxxxxxxxxxxx")

        all_feat_exp = []
        for i in range(0, len(features)):
            if exp_type == "zorro-soft":
                _, feat_exp_i, _ = load_soft_mask(exp_folder, i, "credit")
                # # remove extra dimension
                feat_exp_i = (np.asarray(feat_exp_i)).flatten()

                # # _, feat_exp_i, _ = load_soft_mask(exp_folder, i)
                # # # # remove extra dimension
                # # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                # if device == "cuda":
                #     feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                # else:
                #     feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                # feat_exp_i = feat_exp_i.cpu()
            elif exp_type == "zorro-hard":
                # feat_exp_i = load_minimal_nodes_and_features_sets_zorro(exp_folder, i,
                # check_for_initial_improves=False, isCredit=True)[0][1]
                # # remove extra dimension
                # feat_exp_i = (np.asarray(feat_exp_i)).flatten()
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()
            else:
                if device == "cuda":
                    feat_exp_i = torch.load(exp_folder + str(i))  # load explanations
                else:
                    feat_exp_i = torch.load(exp_folder + str(i), map_location=device)  # load explanations
                feat_exp_i = feat_exp_i.cpu()

            all_feat_exp.append(feat_exp_i)
        # print("all_feat_exp", all_feat_exp)
        # convert list of arrays to single array!
        # if exp_type == "zorro-hard":
        #     all_feat_exp = np.stack(all_feat_exp, axis=0)
        #     all_feat_exp = all_feat_exp.cpu()
        # else:
        all_feat_exp = np.stack(all_feat_exp, axis=0)
        # if exp_type == "gnn-explainer" or exp_type == "grad":  # remove extra dimension
        all_feat_exp = np.squeeze(all_feat_exp) #seems like all of the explanations have extra dim

        # print("features", features) #floating number
        print("features.shape", features.shape) #(3783, 8)

        print(all_feat_exp.shape) #(3783, 8)

        # concert to float tensor
        exp_features = torch.FloatTensor(all_feat_exp)

        # plot_explanations(exp_features, exp_type, data_name, data.y)

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
            print("********************** using explanations with the loss function ********************** ")
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

    G = convert.to_networkx(data, to_undirected=True)
    # print(nx.info(G))
    # print("num classes". max(data.y.numpy()) + 1)
    original_adj = nx.adjacency_matrix(G).todense()

    # Assumption that the attacker can retrive labels from the released model.
    # We used the default features for extracting the labels
    if get_predicted_labels == 1:
        data.y = get_pretrained_labels(path, released_model, data.x, data.edge_index, data.y)

    # return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, original_adj
    if use_exp_with_loss == 1:
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path

    elif get_intersection == 1:
        # Note that to run this, use_defense has to be set
        return original_exp, perturbed_exp

    elif get_fidelity == 1: #get_fidelity
        return explanations, features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, data.edge_index, path
    else:
        # default
        return features, features.shape[1], data.y, max(
            data.y.numpy()) + 1, data.train_mask, data.val_mask, data.test_mask, original_adj, path





















def load_data(args):
    dataset_str = args.dataset

    use_exp = None
    concat_feat_with_exp = None
    exp_only_as_feature = None

    if args.attack_type == "gsef_concat":
        use_exp = True
        concat_feat_with_exp = True
        exp_only_as_feature = False
    elif args.attack_type == "gsef_mult" or args.attack_type == "gsef":
        use_exp = True
        concat_feat_with_exp = False
        exp_only_as_feature = False
    elif args.attack_type == "gse" or args.attack_type == "explainsim":
        use_exp = True
        concat_feat_with_exp = False
        exp_only_as_feature = True
    elif args.attack_type == "slaps" or args.attack_type == "featuresim":
        use_exp = False
        concat_feat_with_exp = False
        exp_only_as_feature = False
    
    


    if dataset_str == "bitcoin":
        nfeats = 8
        nclasses = 2

        released_model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)



        return load_bitcoin('./Dataset/bitcoinalpha.csv', use_exp=use_exp, concat_feat_with_exp=concat_feat_with_exp,
                            exp_only_as_feature=exp_only_as_feature, exp_type=args.explanation_method, 
                            use_exp_with_loss = args.use_exp_as_reconstruction_loss, get_fidelity = args.get_fidelity, 
                            use_defense = args.use_defense, get_intersection=args.get_intersection, 
                            epsilon=args.epsilon, num_exp_in_each_split=args.num_exp_in_each_split, 
                            get_predicted_labels=args.get_predicted_labels, 
                            path = "./saved_models/GCN/Bitcoin_.pth.tar", released_model = released_model)

    elif dataset_str == "credit":
        nfeats = 13
        nclasses = 2

        released_model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)



        return load_credit("./Dataset/Credit/", use_exp=use_exp, concat_feat_with_exp=concat_feat_with_exp,
                            exp_only_as_feature=exp_only_as_feature, exp_type=args.explanation_method,
                            use_exp_with_loss = args.use_exp_as_reconstruction_loss, get_fidelity = args.get_fidelity,
                            use_defense = args.use_defense, get_intersection=args.get_intersection,
                            epsilon=args.epsilon, num_exp_in_each_split=args.num_exp_in_each_split,
                            get_predicted_labels=args.get_predicted_labels,
                            path = "./saved_models/GCN/Creditgcn_2_layers.pt", released_model = released_model)
        # "./saved_models/GCN/Credit_.pth.tar"

    elif dataset_str == "chameleon":
        nfeats = 128
        nclasses = 5

        released_model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)



        return load_chameleon("./Dataset/Chameleon/", use_exp=use_exp, concat_feat_with_exp=concat_feat_with_exp,
                            exp_only_as_feature=exp_only_as_feature, exp_type=args.explanation_method,
                            use_exp_with_loss = args.use_exp_as_reconstruction_loss, get_fidelity = args.get_fidelity,
                            use_defense = args.use_defense, get_intersection=args.get_intersection,
                            epsilon=args.epsilon, num_exp_in_each_split=args.num_exp_in_each_split,
                            get_predicted_labels=args.get_predicted_labels,
                            path = "./saved_models/GCN/Chameleon_.pth.tar", released_model = released_model)

    elif dataset_str == "cora_ml":
        nfeats = 300
        nclasses = 7

        released_model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)

        return load_cora_ml('./Dataset/cora_ml.npz', use_exp=use_exp, concat_feat_with_exp=concat_feat_with_exp,
                            exp_only_as_feature=exp_only_as_feature, exp_type=args.explanation_method,
                            use_exp_with_loss=args.use_exp_as_reconstruction_loss, get_fidelity=args.get_fidelity,
                            use_defense=args.use_defense, get_intersection=args.get_intersection, epsilon=args.epsilon,
                            num_exp_in_each_split=args.num_exp_in_each_split,
                            get_predicted_labels=args.get_predicted_labels, path="./saved_models/GCN/Cora_ml_.pth.tar",
                            released_model=released_model)

    else: #cora, citeseer, pubMed
        nfeats = None
        nclasses = None
        path = None

        if dataset_str == "cora":
            nfeats = 1433
            nclasses = 7
            path = "./saved_models/GCN/Cora_.pth.tar"
        elif dataset_str == "citeseer":
            nfeats = 3703
            nclasses = 6
            path = "./saved_models/GCN/CiteSeergcn_2_layers.pt" #"./saved_models/GCN/CiteSeer_.pth.tar"
        elif dataset_str == "pubmed":
            nfeats = 500
            nclasses = 3
            path = "./saved_models/GCN/Pubmedgcn_2_layers.pt" #"./saved_models/GCN/PubMed_.pth.tar"


        released_model = GCN_PyG(in_channels=nfeats, hidden_channels=args.hidden, out_channels=nclasses,
                                   num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                   sparse=args.sparse)
        # Note to run fidelity only set use_exp=True
        return load_citation_network(dataset_str, use_exp=use_exp, concat_feat_with_exp=concat_feat_with_exp,
                                     exp_only_as_feature=exp_only_as_feature, exp_type=args.explanation_method,
                                     use_exp_with_loss=args.use_exp_as_reconstruction_loss,
                                     get_fidelity=args.get_fidelity, use_defense=args.use_defense,
                                     get_intersection=args.get_intersection, epsilon=args.epsilon,
                                     num_exp_in_each_split=args.num_exp_in_each_split,
                                     get_predicted_labels=args.get_predicted_labels,
                                     path=path, released_model=released_model)
