# modNN
### Modular NN architecture

Simple architecture developed during MScR thesis, allows for definition of experiments using a config file (list of python dictionary entries). Multiple experiments can be run from the command line, or scheduled using a Grid Engine system.

- - - -

Experiments are run by [run.py](src/run.py) using the *run_task* function which task a config dictionary as input.

Example config included at [setup.py](egs/setup.py), there are two formats possible;
* Sequential computational graph (SimpleModel): one input, one output, sequential NN modules
* Customisable computational graph (GraphModel): requires handlers to be given names and an adjacency list to be defined in the config. *Currently, this does no support concatenation of handlers, i.e. no multi-modal support. Splitting of handlers will work, i.e. multi-task learning.*

Example data providers and handlers that subclass from abstract classes are available in the [source](src) directory.

Grid Engine run script included at [run.sh](egs/run.sh), for the son of grid engine system used by our internal GPU cluster.

Plotting functions at [viz.py](src/viz.py), for learning curves and bar charts of multiple experiments.

- - - -

![Class diagram, delegation design pattern](modular-NN-architecture.pdf)
