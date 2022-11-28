import os
import numpy as np
import pandas as pd
import scipy as sc
import pandas as pd

import numpy as np
import networkx as nx
import abc


class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        # Normalize feature
        feat = (feat+np.max(np.abs(feat)))/np.max(np.abs(feat))/2
        feat_dict = {
                i: {"feat": feat[i]} for i in range(feat.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)


def read_bitcoinalpha(dataset, feature_generator=None):

    df = pd.read_csv(dataset)

    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)

    mapping = {}
    count = 0
    for node in list(G.nodes):
        count = count + 1
        mapping[node] = count
    G = nx.relabel_nodes(G, mapping)

    rating = nx.get_edge_attributes(G, 'RATING')
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)

    label_mapping = {}
    rate_mapping = {}
    decision_threshold = 0.3
    number_of_in_nodes_threshold = 3

    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        if len(in_edges_list) < number_of_in_nodes_threshold:
            total_rate = 0
            label = 0
            rate_mapping[node] = 0
            label_mapping[node] = label
        else:
            total_rate = 0
            for (source, _) in in_edges_list:
                total_rate = total_rate + G.get_edge_data(source, node)['RATING'] / np.abs(
                    G.get_edge_data(source, node)['RATING'])
            average_rate = total_rate / len(in_edges_list)

            label = 0
            if average_rate < decision_threshold:
                label = 0
            else:
                label = 1

            rate_mapping[node] = average_rate
            label_mapping[node] = label

    roles = []
    count = 0
    count1 = 0
    for node, l in label_mapping.items():
        count = count + 1
        if l == 1:
            count1 = count1 + 1
        roles.append(l)
    print("Total node: ", count)
    print("Positive node: ", count1)

    if feature_generator is None:

        feat_dict = {}
        feature_length = 8
        for node in list(G.nodes):
            out_edges_list = G.out_edges(node)

            if len(out_edges_list) == 0:
                features = np.ones(feature_length, dtype=float) / 1000
                feat_dict[node] = {'x': features}
            else:
                features = np.zeros(feature_length, dtype=float)
                w_pos = 0
                w_neg = 0
                for (_, target) in out_edges_list:
                    w = G.get_edge_data(node, target)['RATING']
                    if w >= 0:
                        w_pos = w_pos + w
                    else:
                        w_neg = w_neg - w

                abstotal = (w_pos + w_neg)
                average = (w_pos - w_neg) / len(out_edges_list) / max_rating

                features[0] = w_pos / max_rating / len(out_edges_list)  # average positive vote
                features[1] = w_neg / max_rating / len(out_edges_list)  # average negative vote
                features[2] = w_pos / abstotal
                features[3] = average
                features[4] = features[0] * G.in_degree(node) / dmax_in
                features[5] = features[1] * G.in_degree(node) / dmax_in
                features[6] = features[0] * G.out_degree(node) / dmax_out
                features[7] = features[1] * G.out_degree(node) / dmax_out

                features = features / 1.01 + 0.001

                feat_dict[node] = {'x': features}
        print("Good nodes ratio: ", count1 / count)

        nx.set_node_attributes(G, feat_dict)
    else:
        feature_generator.gen_node_features(G)

    name = "bitcoinalpha"
    G = G.to_undirected()

    return G, roles, name


