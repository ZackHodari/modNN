import tensorflow as tf
from src.load import DemoProvider
from src.input_handlers import *
from src.module_handlers import *
from src.output_handlers import *


tasks = {
    'test_simple': [
        {
            'name': 'test/simple',
            'data_provider': DemoProvider,
            'input_handlers': Waves,
            'module_handlers': [FCModule(layer_dims=[20], nonlinearity=tf.nn.relu)],
            'output_handlers': WaveClasses
         },
    ],

    'test_graph': [
        {
            'name': 'test/graph',
            'data_provider': DemoProvider,
            'input_handlers': {'input 1': Waves},
            'module_handlers': {'module 1': FCModule(layer_dims=[20], nonlinearity=tf.nn.relu)},
            'output_handlers': {'output 1': WaveClasses},
            'graph': [('input 1', ['module 1']), ('module 1', ['output 1'])]
         },
    ],
}



