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
    parser.add_argument('-m', '--max-out-nodes', type=int, default=5)
    parser.add_argument('-z', '--horizontal', action='store_true')
    return parser.parse_args()

def run():
    args = get_args()
    with open(args.model) as model_file:
        model = json.load(model_file)
    dot = model_to_dot(model, rankdir='LR' if args.horizontal else 'TB')
    add_outputs(dot, model, args.max_out_nodes)
    dot.write(str(args.output), format=args.output.suffix.lstrip('.'))

def comps_as_string(layer):
    comps = layer['components']
    dims = [(c,len(x['bias'])) for c, x in comps.items()]
    assert all(d[1] == dims[0][1] for d in dims)
    return str(dims[0][1])

def str_from_activation(layer):
    act = layer['activation']
    try:
        return act['function']
    except TypeError:
        return act

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
                    label += '\n {}'.format(str_from_activation(layer))
                else:
                    label = str_from_activation(layer)
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

    return dot

def add_outputs(dot, model, max_outputs):

    # add outputs
    for node_name, output_node in model['outputs'].items():
        source = output_node['node_index']
        if len(output_node['labels']) > max_outputs:
            lab = node_name
            num = len(output_node['labels'])
            out_name = f'out_{source}_{lab}'
            dot.add_node(pydot.Node(out_name, label=lab + f'({num})'))
            dot.add_edge(pydot.Edge(str(source), out_name))
        else:
            for lab in output_node['labels']:
                out_name = f'out_{source}_{lab}'
                dot.add_node(pydot.Node(out_name, label=lab))
                dot.add_edge(pydot.Edge(str(source), out_name))

    return dot

if __name__ == '__main__':
    run()
