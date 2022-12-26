import dash_cytoscape as cyto
from dash import Dash, html, Input, Output, ctx, State
from tree_visualization.stylesheets import tree_stylesheet
from tree_visualization.element_generator import generate_elements, get_default_elements, get_subtree_nodes, get_subtree_edges

# enable svg export
cyto.load_extra_layouts()


def get_tree_visualization(root, depth=1):
    """
    Main function which can be called to get a visualization app for a given tree
    :param root: root of the tree which will be visualized
    :param depth: tree depth before expansion
    """
    app = Dash(__name__)
    server = app.server

    # Visualization elements generation
    tree = root
    nodes, edges = generate_elements(tree)
    default_elements = get_default_elements(nodes, edges, depth=depth)
    subtree_nodes = get_subtree_nodes(nodes)

    # App layout
    layout = {'name': 'breadthfirst',
              'roots': '[id = "root"]'}

    app.layout = html.Div([
        html.Div(
            cyto.Cytoscape(
                id='cytoscape-tree',
                elements=default_elements,
                stylesheet=tree_stylesheet,
                layout=layout,
                style={
                    'height': '90vh',
                    'width': '100%',
                    'float': 'top'
                }
            )),
        html.Div(children=[
            html.Div('Download graph:'),
            html.Button("as jpg", id="btn-get-jpg"),
            html.Button("as png", id="btn-get-png"),
            html.Button("as svg", id="btn-get-svg")
        ]),
        html.Div(html.P(
            id='hover-output'
        ))
    ])

    # Callbacks
    @app.callback(
        Output("cytoscape-tree", "generateImage"),
        [
            Input("btn-get-jpg", "n_clicks"),
            Input("btn-get-png", "n_clicks"),
            Input("btn-get-svg", "n_clicks"),
        ])
    def get_image(get_jpg_clicks, get_png_clicks, get_svg_clicks):
        # Default behavior if no button is clicked
        ftype = 'png'
        action = 'store'

        # When a button is clicked, download image in the corresponding format
        if ctx.triggered:
            action = "download"
            ftype = ctx.triggered_id.split("-")[-1]

        return {
            'type': ftype,
            'action': action
        }

    @app.callback(Output('cytoscape-tree', 'elements'),
                  Input('cytoscape-tree', 'tapNodeData'),
                  State('cytoscape-tree', 'elements'))
    def expand_node(nodeData, elements):
        if not nodeData:
            return default_elements

        # Define the nodes and edges of the subtree rooted at the given node
        following_nodes = subtree_nodes[nodeData['id']]
        following_edges = get_subtree_edges(following_nodes, edges)

        if nodeData['expandable']:
            # Node not yet expanded
            # Expand the subtree rooted at the selected node
            for node in following_nodes:
                if node not in elements:
                    elements.append(node)
                    node['data']['expanded'] = True
            for edge in following_edges:
                if edge not in elements:
                    elements.append(edge)
        else:
            # Node already expanded
            # Compress the subtree rooted at the selected node
            for node in following_nodes:
                if node in elements:
                    elements.remove(node)
                    node['data']['expanded'] = False
            for edge in following_edges:
                if edge in elements:
                    elements.remove(edge)

        # Update the node's data
        for element in elements:
            if nodeData['id'] == element['data']['id']:  # Identify the node
                element['data']['expandable'] = not element['data']['expandable']
                break

        return elements

    @app.callback(Output('hover-output', 'children'),
                  Input('cytoscape-tree', 'mouseoverNodeData'))
    def display_hover_node_data(data):
        if data:
            return 'expand' if data['expandable'] else 'compress'

    # Run the app
    app.run_server(debug=True)

# TODO fix expansion status inconsistencies in subtrees (use counter?)
# TODO show if a node is expandable (through layout instead of callback? node opacity)
