import dgl
import torch
from collections import defaultdict
import numpy as np
import config

def load_features(features, edges, num_features):
    num_nodes = 100386
    num_feats = num_features
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open(features) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    ret_edges = []
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            ret_edges.append((paper1 , paper2))
            ret_edges.append((paper2 , paper1))

    U , V = zip(*ret_edges)
    ret_edges = (list(U) , list(V))
    print(label_map)
    return feat_data, labels, ret_edges

def get_annotated_users(mode):
    import pandas as pd
    if mode == "hate":
        df = pd.read_csv(config.ACTIVITY_CSV)
        df = df[df.hate != "other"]
        y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])
        x = np.array(df["user_id"].values)
        del df

    else:
        df = pd.read_csv(config.ACTIVITY_CSV)
        np.random.seed(321)
        df2 = df[df["is_63_2"] == True].sample(668, axis=0)
        df3 = df[df["is_63_2"] == False].sample(5405, axis=0)
        df = pd.concat([df2, df3])
        y = np.array([1 if v else 0 for v in df["is_63_2"].values])
        x = np.array(df["user_id"].values)
        del df, df2, df3

    return x