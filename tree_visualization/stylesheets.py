# Stylesheet used for the tree visualization
# Describes the appearance of the tree's nodes and edges
tree_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'shape': 'round-rectangle',
            'width': 100,
            'height': 80,
            'background-color': 'white',
            'border-width': 1,
            'border-color': 'black',
            'content': 'data(info_string)',
            'text-wrap': 'wrap',
            'text-max-width': 78,
            'text-halign': 'center',
            'text-valign': 'center'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'label': 'data(action)',
            'text-wrap': 'wrap',
            'text-margin-x': -30,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            "source-endpoint": "outside-to-node",
            "target-endpoint": "outside-to-node",
        }
    }
]
