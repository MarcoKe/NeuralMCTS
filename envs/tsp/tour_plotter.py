import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import dgl

def plot_state(tour, figsize=5, show_axes=True, color='r', edge_alpha=None, node_size=300, line_width=1, pad=0.1):
    graph = nx.complete_graph(len(tour))
    traversed_edges = [(i, i + 1) for i in range(len(tour) - 1)]
    traversed_edges.append((len(tour) - 1, 0))
    # plot setup
    positions = tour
    padding = pad / figsize
    map_size = 1  # self.__map_size() TODO
    # visited_nodes = np.where(self.state.ndata['visited'] == 1)[0].tolist()
    # unvisited_nodes = np.where(
    #     self.state.ndata['visited'] == 0)[0].tolist()

    fig, ax = plt.subplots(figsize=(figsize, figsize), constrained_layout=True)

    # all edges
    nx.draw_networkx_edges(graph, ax=ax, pos=positions, width=line_width,
                           alpha=(0.5 / (len(tour) ** 0.5) if edge_alpha is None else edge_alpha))
    # traversed edges
    nx.draw_networkx_edges(graph, ax=ax, edgelist=traversed_edges,
                           pos=positions, width=2 * line_width, edge_color=color)
    # visited nodes
    nx.draw_networkx_nodes(graph, ax=ax, pos=positions,
                           node_shape='.', node_color=color, node_size=node_size)
    # # unvisited_nodes
    # nx.draw_networkx_nodes(graph, ax=ax, nodelist=unvisited_nodes, pos=positions,
    #                        node_shape='.', node_size=node_size)
    # start node
    nx.draw_networkx_nodes(graph, ax=ax, nodelist=[0], pos=positions,
                           node_shape='.', node_color='k', node_size=node_size)

    if show_axes:
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
    else:
        ax.set_axis_off()
    limits = (-padding * map_size, (padding + 1) * map_size)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    plt.show()

    return fig, ax


def build_graph(positions):
    # graph setup
    G = nx.Graph()
    graph = nx.complete_graph(len(positions))
    # graph = dgl.from_networkx(graph)

    # node and edge features
    graph.nodes.data['pos'] = positions
    # graph.edata['dist'] = distances

    graph = graph.add_self_loop()  # for graph convolutions

    # node status indicators
    graph.nodes[0].data['visited'] = 1
    graph.nodes[0].data['current'] = 1
    graph.nodes[0].data['start'] = 1

    return graph

if __name__ == '__main__':
    tour = [(0.3, 0.6), (0.6, 0.5), (0.1, 0.1), (0.2, 0.3)]


    print(traversed_edges)
    plot_state(G, traversed_edges, tour)

    # nx.draw_networkx_nodes(G, pos=tour)
    #
    # a = nx.draw_networkx_edges(G, pos=tour, edge_cmap=plt.cm.magma)
    # a = nx.draw_networkx_edges(G, pos=tour, edge_color='red', edgelist=traversed_edges)
    #
    # plt.show()
    # build_graph(tour)