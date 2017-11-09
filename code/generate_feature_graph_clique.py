
# coding: utf-8

import networkx as nx
import pandas as pd
from itertools import combinations


df_train = pd.read_csv('../../input/train.csv').fillna("")
df_test = pd.read_csv('../../input/test.csv').fillna("")
len_train = df_train.shape[0]

df = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]], axis=0)


G = nx.Graph()
edges = [tuple(x) for x in df[['question1', 'question2']].values]
G.add_edges_from(edges)

map_label = dict(((x[0], x[1])) for x in df[['question1', 'question2']].values)
map_clique_size = {}
cliques = sorted(list(nx.find_cliques(G)), key=lambda x: len(x))
for cli in cliques:
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_label:
            map_clique_size[q1, q2] = len(cli)
        elif (q2, q1) in map_label:
            map_clique_size[q2, q1] = len(cli)

df['clique_size'] = df.apply(lambda row: map_clique_size.get((row['question1'], row['question2']), -1), axis=1)


df[['clique_size']][:len_train].to_csv('train_feature_graph_clique.csv', index=False)
df[['clique_size']][len_train:].to_csv('test_feature_graph_clique.csv', index=False)




