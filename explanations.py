import numpy as np
import torch

# print(torch.__version__)
# exit()
from torch_geometric.nn import MessagePassing
import os
import torch.nn.functional as F
import torch_geometric.utils as ut
from pathlib import Path
from torch_geometric.utils import to_dense_adj,k_hop_subgraph
def save_model(model, path):
    torch.save(model.state_dict(), path)
from torch_geometric.utils import to_dense_adj,k_hop_subgraph
from graph_lime import *
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from models import load_dataset,GCNNet, load_model
from models import *
from grad_explainer import *
import collections
from fidelity import *
from gnn_explainer import *
from zorro import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
import argparse
from scipy.stats import entropy


def arg_parse():
    parser = argparse.ArgumentParser(description="arguments for generating explanations")
    parser.add_argument("--dataset",type=str,dest="dataset",help="input dataset")
    parser.add_argument("--explainer",type=str,dest="explainer",help="explainer to be used")
    parser.add_argument("--model",type=str,dest="model",help="GNN model")
    parser.add_argument("--save_exp",action="store_true",default=False,help="save explanation")
    parser.add_argument("--untrained_gnn", action="store_true", default=False,
                        help="use the untrained GNN")
    parser.add_argument("--start",type=int, dest="start",help="start node")
    parser.add_argument("--end",type=int, dest="end",help="end node")
    parser.set_defaults(dataset="Cora",
                        explainer="Grad",
                        model="GCN")

    return parser.parse_args()

args = arg_parse()

working_directory = Path("/home/rathee/privacy/").resolve()

data_set = args.dataset
dataset, data, results_path = load_dataset(data_set, working_directory=working_directory)
data.to(device)

model_dict = {"GCN":GCNNet,"GAT":GATNet,"GIN":GINConvNet,"APPNP":APPNP2Net}
explainer_dict ={"Grad":grad_node_explanation,"GradInput":gradinput_node_explanation,"GraphLime":GLIME,"GNNExplainer":GNNExplainer}

model = model_dict[args.model](dataset)
model.to(device)
save_dir = 'saved_models'
model_directory = args.model
filename = os.path.join(save_dir, args.model)
saved_model = os.path.join(filename,args.dataset)
saved_model_dir = saved_model+"_.pth.tar"

load_model(saved_model_dir,model)

#### if zorro model is used
#load_model("/home/rathee/privacy/Saved_Explanations/hard_zorro/GCN/Cora/gcn_2_layers.pt",model)



if args.untrained_gnn:
    stat_dict = torch.load(saved_model_dir)
    conv1 = stat_dict['conv1.weight']
    bias1 = stat_dict['conv1.bias']
    conv2 = stat_dict['conv2.weight']
    bias2 = stat_dict['conv2.bias']

    #load_model(path=saved_model_dir,model=model)
    model.to(device)

    new_stat_dict = collections.OrderedDict()
    new_stat_dict['conv1.weight'] = torch.ones(conv1.shape)
    new_stat_dict['conv1.bias'] = torch.ones(bias1.shape)
    new_stat_dict['conv2.weight'] = torch.ones(conv2.shape)
    new_stat_dict['conv2.bias'] = torch.ones(bias2.shape)

    #new_stat_dict.to(device)
    model.load_state_dict(new_stat_dict)

    model.to(device)



def subgraph(model, node_idx, x, edge_index, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    flow = 'source_to_target'
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=num_nodes, flow=flow)

    x = x[subset]
    for key, item in kwargs:
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]

        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, kwargs

num_nodes, num_features = data.x.size()



selected_nodes=[0,1,2,3]
print(model)
Fidelity =[]
Sparsity =[]
Fidelity_edge =[]

exp_save_dir = 'Saved_Explanations'
if args.untrained_gnn:
    model_directory="Untrained_GCN"
else:
    model_directory = args.model
explainer = args.explainer
exp_filename = os.path.join(exp_save_dir, args.explainer)
exp_filename = os.path.join(exp_filename, model_directory)
exp_dir = os.path.join(exp_filename,data_set)
print(exp_dir,"explanation")
for node in range(num_nodes):  # 334, 400
    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node, data.x, data.edge_index)
    computation_graph_edge_index.to(device)
    computation_graph_feature_matrix.to(device)

    computation_data = Data(x=computation_graph_feature_matrix,
                            edge_index=computation_graph_edge_index).to(device)
    computation_data.to(device)
    if args.explainer=="Grad":
        #node = torch.tensor(node).to(device)
        feature_mask, node_mask = grad_node_explanation(model,mapping,computation_graph_feature_matrix,computation_graph_edge_index)
        feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)

    if args.explainer=="GradInput":
        feature_mask, node_mask = gradinput_node_explanation(model,mapping,computation_graph_feature_matrix,computation_graph_edge_index)
        feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)
    if args.explainer=="GraphLime":
        lime = GLIME(model,computation_data.x,mapping,computation_data.edge_index,2,rho=0.1,device=device)##0.32
        feature_mask= lime.explain(x=data.x)
        feature_mask = feature_mask.reshape(1,-1).to(torch.float32).to(device)
    if args.explainer=="GNNExplainer":
        gnn_explainer = GNNExplainer(model,log=False)
        feature_mask,edge_mask = gnn_explainer.explain_node(node_idx=mapping,x=computation_graph_feature_matrix,edge_index=computation_graph_edge_index)
        edge_mask  =torch.nn.Parameter(edge_mask)
        feature_mask = feature_mask.reshape(1, -1)
        # fid_edge = fidelity(model,node,full_feature_matrix=data.x,edge_index=data.edge_index,feature_mask=feature_mask,edge_mask=edge_mask)
        # Fidelity_edge.append(fid_edge)
    if args.explainer=="hard_zorro":
        hard_zorro = SISDistortionGraphExplainer(model,device)
        explanation = hard_zorro.explain_node(node,data.x,data.edge_index,tau=0.03,recursion_depth=1)
        selected_nodes, feature_mask, executed_selections = explanation[0]
        feature_mask = torch.from_numpy(feature_mask).to(device)
    if args.explainer=="soft_zorro":
        soft_zorro= GradientBasedDistortionGraphExplainer(model,device)
        explanation = soft_zorro.explain_node(node,data.x,data.edge_index)
        node_masks= explanation[0]
        feature_mask = explanation[1].reshape(1,-1)
        feature_mask = torch.from_numpy(feature_mask).to(device)
    if args.explainer=="None":
        # file_path = exp_dir+"/feature_masks_node="+str(node)
        # feature_mask  = torch.load(file_path)

        file_path = exp_dir+ "/gcn_2_layers_explanation_node_"+str(node)+".npz"
        #"/home/rathee/privacy/Saved_Explanations/hard_zorro/GCN/cora_ml/gcn_2_layers_explanation_t_3_r_1_node_0.npz"
        save = np.load(file_path)

        print(node)

        # for k in save.files:
        #     print(k)
        #     print(save["feature_mask"])
        # feature_mask = save["features_0"]
        feature_mask = save["feature_mask"]



        feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)

    fid = fidelity(model,node,full_feature_matrix=data.x,edge_index=data.edge_index,feature_mask=feature_mask,device=device)
    #edge_mask = edge_mask.to(device).to(torch.float32)
    #sprs = entropy(feature_mask.cpu())
    if feature_mask.nonzero().shape[0]==0:
        sprs=0
    else:
        sprs = entropy(feature_mask.squeeze().cpu())
    if sprs ==-float('inf'):
        sprs=0
    Fidelity.append(fid)
    Sparsity.append(sprs)
    file_path = exp_dir+"/feature_masks_node="+str(node)
    file_path_edge = exp_dir+"/edge_masks_node="+str(node)
    if args.save_exp:
        torch.save(feature_mask, file_path)
        #torch.save(edge_mask, file_path_edge)

# explanation_path = "/home/rathee/privacy/Saved_Explanations/Graph_Lime_Cora_rho=0.01/"
#
# for node in range(num_nodes):
#     computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
#             subgraph(model, node, data.x, data.edge_index)
#     computation_graph_edge_index.to(device)
#     computation_graph_feature_matrix.to(device)
#
#     computation_data = Data(x=computation_graph_feature_matrix,
#                                 edge_index=computation_graph_edge_index).to(device)
#     lime = GLIME(model,computation_data.x,mapping,computation_data.edge_index,2,rho=0.01,device=device)##0.32
#     feature_mask= lime.explain(x=data.x)
#     print(feature_mask.nonzero())
#     file_path = explanation_path+"feature_masks_node="+str(node)
#     torch.save(feature_mask, file_path)
#     print("#"*100)

print(dataset)
print("Fidelity:",np.mean(Fidelity))
print("Sparsity",np.mean(Sparsity))



