#!/usr/bin/env python3

"""
Draw a model from lwtnn
"""

_epilogue = """NOTE:
Right now this only supports the graph models, but you can convert
sequential models to graph models with `sequential2graph.py`
"""

import pydot_ng as pydot
from argparse import ArgumentParser
from pathlib import Path
import json

def get_args():
    parser = ArgumentParser(description=__doc__, epilog=_epilogue)
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

def comps_as_string(layer):
    comps = layer['components']
    dims = [(c,len(x['bias'])) for c, x in comps.items()]
    assert all(d[1] == dims[0][1] for d in dims)
    return str(dims[0][1])

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
        shape = 'ellipse'
        if layer_type in ['feed_forward', 'time_distributed', 'sequence']:
            layer = layers[lwtnn_node['layer_index']]
            label = layer['architecture']
            if label == 'dense':
                if layer['bias']:
                    label += ' ({})'.format(len(layer['bias']))
                    label += '\n {}'.format(layer['activation'])
                else:
                    label = layer['activation']
            elif label in ['lstm', 'gru']:
                label += ' ({})'.format(comps_as_string(layer))
        elif layer_type == 'input':
            shape = 'rectangle'
            in_node = model['inputs'][lwtnn_node['sources'][0]]
            label = '{}\n({})'.format(
                in_node['name'], len(in_node['variables']))
        elif layer_type == 'input_sequence':
            shape = 'rectangle'
            in_node = model['input_sequences'][lwtnn_node['sources'][0]]
            label = '{}\n({})'.format(
                in_node['name'], len(in_node['variables']))
        else:
            label = layer_type
        dot_node = pydot.Node(str(num), label=label, shape=shape)
        dot.add_node(dot_node)

    # Connect nodes with edges.
    for dest_node, node in enumerate(nodes):
        if node['type'] in ['input_sequence', 'input']:
            continue
        for source_node in node['sources']:

            dot.add_edge(pydot.Edge(str(source_node), str(dest_node)))

    # add outputs
    for output_node in model['outputs'].values():
        source = output_node['node_index']
        for lab in output_node['labels']:
            out_name = f'out_{source}_{lab}'
            dot.add_node(pydot.Node(out_name, label=lab))
            dot.add_edge(pydot.Edge(str(source), out_name))

    return dot

if __name__ == '__main__':
    run()
