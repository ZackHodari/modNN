from egs.setup import tasks
from models import SimpleModel, GraphModel
import sys
import os

num_epochs = 40


"""
USAGE
Through Son of Grid Engine:

- Use submit.sh:
bash submit.sh <job_name> <task_set> <begin_id> <end_id>

- Or, single job:
qsub -N <job_name> run.sh <task_set> <task_id>

- Or, array job:
qsub \
  -N <job_name> \
  -t <begin_id>-<end_id>:1 \
  -tc <concurrent_jobs> \
  run.sh <task_set>  
"""


# retrieve the task set and output type from command line argument
task_set = sys.argv[1]  # baseline_FC | baseline_RNN | tests
assert task_set in tasks.keys(), (
    'task_set must be a key in tasks, got {}'.format(task_set))

# set task_id and get task definition, using array job id or command line argument
if 'SGE_TASK_ID' in os.environ and os.environ['SGE_TASK_ID'] != 'undefined':
    task_id = int(os.environ['SGE_TASK_ID'])
else:
    task_id = int(sys.argv[2])

assert 0 < task_id <= len(tasks[task_set]), (
    'task_id must be between 1 and {}, got {}'.format(len(tasks[task_set]), task_id))

# get the individual task to be run
task = tasks[task_set][task_id - 1]

# set output path and ensure a directory exists for this path
output_path = os.path.join('results', task['name'])
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# initialise the model
if 'graph' in task:
    # use the given graph structure
    model = GraphModel(task['name'], task['data_provider'],
                       task['input_handlers'], task['module_handlers'], task['output_handlers'], task['graph'],
                       add_summaries=True)
else:
    # no graph structure given, create simple chain graph
    model = SimpleModel(task['name'], task['data_provider'],
                        task['input_handler'], task['model_handlers'], task['output_handler'],
                        add_summaries=True)

# report the model built
print('task_set: {}\ntask_id: {}\ntask_name: {}\n'.format(task_set, task_id, task['name']))
print(model)

# reload or train the model
if os.path.isfile(os.path.join('results', model.experiment_name, 'model', 'trained_model.ckpt.index')):
    # load the model
    model.restore_model()
else:
    # train the model
    model.train(num_epochs=num_epochs)



