import networkx as nx
from  fast_unfolding import *

from collections import defaultdict

def makeSampleGraph():
    """
    :return:    获取一个图
    """
    g = nx.Graph()
    g.add_edge("a", "b", weight=1.)
    g.add_edge("a", "c", weight=1.)
    g.add_edge("b", "c", weight=1.)
    g.add_edge("b", "d", weight=1.)
    # 图结构如下：
    # {'a'：{'c': {'weight': 1.0}, 'b': {'weight': 1.0}}}
    return g

if __name__ == "__main__":
    sample_graph = makeSampleGraph()
    print(sample_graph.nodes,sample_graph.edges)
    print(sample_graph['a'])
    louvain = Louvain()
    partition = louvain.getBestPartition(sample_graph)

    p = defaultdict(list)
    for node, com_id in partition.items():
        p[com_id].append(node)

    for com, nodes in p.items():
        print(com, nodes)