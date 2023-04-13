def generate_elements(root, grabbable=False):
    """
    Generates nodes and edges for the visualization given a tree root
    :param root: root node of the tree structure which will be visualized
    :return nodes: node elements used by Dash for the visualization
    :return edges: edge elements used by Dash for the visualization
    """
    def add_to_elements(node, node_id):
        children = node.children

        node_source = {
            "data": {
                'id': node_id,
                'visits': node.visits,
                'avgReturn': node.returns / node.visits,
                'maxReturn': node.max_return,
                'info_string': f"visits: {node.visits:.2f}\n returns: {node.returns:.2f} \n avg_return: {(node.returns/node.visits):.2f}\nmax_return: {node.max_return:.3f}",
                'expandable': False,
                'depth': get_depth(root, node),
                'leaf': node.is_leaf()
            },
            'classes': 'node',
            'grabbable': grabbable
        }
        nodes.append(node_source)

        for n, child in enumerate(children):
            child_id = node_id + 'c' + str(n)

            # Add an edge between the child node and its parent
            node_edge = {
                'data': {
                    'id': node_id + '-' + child_id,
                    'source': node_id,
                    'target': child_id,
                    'sourceNodeId': node_id,
                    'action': child.action
                }
            }
            edges.append(node_edge)

            # Recursively traverse the child nodes
            add_to_elements(child, child_id)

    nodes = []
    edges = []

    add_to_elements(root, 'root')

    return nodes, edges


def get_depth(root, node):
    """
    Finds the depth of a given node
    :param root: the root of the tree containing the node
    :param node: the node whose depth is calculated
    :return: the depth of the node
    """
    d = 0

    # Count the nodes from the given node up to the root of the tree
    while node is not root:
        d += 1
        node = node.parent

    return d


def get_default_elements(nodes, edges, depth=0):
    """
    Defines the elements shown before any node expansion
    :param nodes: nodes used by Dash for the visualization
    :param edges: edges used by Dash for the visualization
    :param depth: tree depth before expansion
    :return: list of nodes and edges shown before node expansion
    """
    # Get the nodes up to depth d
    default_nodes = []
    for n in nodes:
        if n['data']['depth'] <= depth:
            default_nodes.append(n)
            # The last level of the default nodes can be expanded, unless the nodes are leaves
            if n['data']['depth'] == depth and not n['data']['leaf']:
                n['data']['expandable'] = True

    # Get the edges connecting nodes up to depth d
    default_node_ids = [n['data']['id'] for n in default_nodes]
    default_edges = [e for e in edges if e['data']['target'] in default_node_ids]

    return default_nodes + default_edges


def get_subtree_nodes(nodes):
    """
    Generates dictionaries containing the nodes and edges in each subtree (needed for node expansion)
    :param nodes: nodes used by Dash for the visualization
    :return subtree_nodes: dictionary with node id as keys and lists of the corresponding subtree nodes as values
    """
    def get_descendants(node):
        children = [c for c in nodes if (node['data']['id'] + 'c') in c['data']['id']]
        return children

    subtree_nodes = {n['data']['id']: get_descendants(n) for n in nodes}
    return subtree_nodes


def get_subtree_edges(nodes, edges):
    """
    Returns the edges between given nodes
    :param nodes: node subset for which the edges are returned
    :param edges: all existing edges
    :return: edges between the given nodes
    """
    ids = [n['data']['id'] for n in nodes]
    subtree_edges = [e for e in edges if e['data']['target'] in ids or e['data']['source'] in ids]
    return subtree_edges
