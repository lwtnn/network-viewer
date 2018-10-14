#!/usr/bin/env python3

"""
Draw a model from lwtnn
"""

import pydot_ng as pydot
from argparse import ArgumentParser
from pathlib import Path
import json

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('-o','--output', default='model_graph.pdf',
                        type=Path)
    return parser.parse_args()

def run():
    args = get_args()
    with open(args.model) as model_file:
        model = json.load(model_file)
    dot = model_to_dot(model)
    dot.write(str(args.output), format=args.output.suffix.lstrip('.'))

def model_to_dot(model,
                 rankdir='TB'):
    """I stole this from Keras, then hacked

    Returns:
        A `pydot.Dot` instance representing the lwtnn model.
    """

    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    nodes = model['nodes']
    layers = model['layers']
    # Create graph nodes.
    for num, lwtnn_node in enumerate(nodes):
        layer_type = lwtnn_node['type']
        if layer_type in ['feed_forward', 'time_distributed', 'sequence']:
            layer = layers[lwtnn_node['layer_index']]
            label = layer['architecture']
            if label == 'dense':
                label += '\n {}'.format(layer['activation'])
        elif layer_type == 'input':
            in_node = model['inputs'][lwtnn_node['sources'][0]]
            label = '{}\n({})'.format(
                in_node['name'], len(in_node['variables']))
        elif layer_type == 'input_sequence':
            in_node = model['input_sequences'][lwtnn_node['sources'][0]]
            label = '{}\n({})'.format(
                in_node['name'], len(in_node['variables']))
        else:
            label = layer_type
        dot_node = pydot.Node(str(num), label=label)
        dot.add_node(dot_node)

    # Connect nodes with edges.
    for dest_node, node in enumerate(nodes):
        if node['type'] in ['input_sequence', 'input']:
            continue
        for source_node in node['sources']:

            dot.add_edge(pydot.Edge(str(source_node), str(dest_node)))
    return dot

if __name__ == '__main__':
    run()
