import numpy as np

DEFAULT_SEED = 1234567890


class AbstractProvider(object):
    """
    Abstract data provider, uses a config that is passed onto the AbstractSample class which loads the datum

    :param file_paths: list - file paths which samples will use to load the data
    :param data_config: dictionary - configuration specifying what to load for each sample
    :param batch_size: integer - how big each mini-batch should be
    :param shuffle_data: boolean - shuffle the training data
    :param rng_seed: integer (optional) - custom seed for rng used for shuffling
    """

    def __init__(self, file_paths, data_config, batch_size, shuffle_data, rng_seed=DEFAULT_SEED, data_splitter=None):
        assert isinstance(file_paths, list), (
            'file_paths must be a list'
            '\nGot {}'.format(type(file_paths)))
        assert isinstance(file_paths, list), (
            'file_paths must be non-empty'
            '\nGot {}'.format(len(file_paths)))
        assert isinstance(data_config, dict), (
            'data_config must be a dictionary'
            '\nGot {}'.format(type(data_config)))
        assert isinstance(rng_seed, int), (
            'rng_seed must be an integer'
            '\nGot {} ({})'.format(rng_seed, type(rng_seed)))
        assert isinstance(batch_size, int), (
            'batch_size must be an integer'
            '\nGot {}'.format(type(batch_size)))
        assert batch_size > 0, (
            'batch_size must be greater than zero'
            '\nGot {}'.format(batch_size))
        assert isinstance(shuffle_data, bool), (
            'shuffle_data must be a boolean'
            '\nGot a {}'.format(type(shuffle_data)))

        self.file_paths_all = np.array(file_paths)
        self.rng_seed       = rng_seed
        self.rng            = np.random.RandomState(rng_seed)

        if data_splitter is None:
            self.train_valid_test_split()
        else:
            # use a custom splitting function instead
            data_splitter()

        self._current_order = np.arange(self.file_paths_train.shape[0])
        self.data_config    = data_config
        self.batch_size     = batch_size
        self.shuffle_data   = shuffle_data

    # Splits the file paths into 3 lists for training, validation and testing
    def train_valid_test_split(self, valid_size=0.15, test_size=0.15):
        num_samples = self.file_paths_all.shape[0]
        indices = self.rng.permutation(num_samples)

        test_index = int(num_samples*test_size)
        valid_index = int(num_samples*valid_size) + test_index

        # data = [test_data, valid_data, train_data]
        self.file_paths_test  = self.file_paths_all[indices[0:           test_index]]
        self.file_paths_valid = self.file_paths_all[indices[test_index:  valid_index]]
        self.file_paths_train = self.file_paths_all[indices[valid_index: None]]

    # Resets the data provider to the initial state, resetting the rng and fixing the order of the training data
    def reset(self):
        self.rng = np.random.RandomState(self.rng_seed)
        inv_perm = np.argsort(self._current_order)

        self._current_order = self._current_order[inv_perm]
        self.file_paths_train = self.file_paths_train[inv_perm]

    # Randomly shuffles the training data
    def shuffle(self):
        perm = self.rng.permutation(self.file_paths_train.shape[0])
        self._current_order = self._current_order[perm]
        self.file_paths_train = self.file_paths_train[perm]

    # Given a list of file paths, yield mini-batches (AbstractBatch) of self.batch_size samples (AbstractSample)
    def yield_batches(self, file_paths):
        num_batches = int((file_paths.shape[0] - 1) // self.batch_size + 1)

        for batch_num in range(num_batches):
            batch_slice = slice(self.batch_size * batch_num,
                                self.batch_size * (batch_num + 1))

            yield self.load_data(file_paths[batch_slice])

    def load_data(self, file_paths):
        # e.g. return AbstractBatch(file_paths, self.data_config)
        raise NotImplementedError('This should be implemented in subclasses for specific datasets')

    # A generator for batches from the test data
    @property
    def train_data(self):
        if self.shuffle_data:
            self.shuffle()

        return self.yield_batches(self.file_paths_train)

    @property
    def valid_data(self):
        return self.yield_batches(self.file_paths_valid)

    @property
    def test_data(self):
        return self.yield_batches(self.file_paths_test)

    def all_data(self):
        return self.yield_batches(self.file_paths_all)

    def __iter__(self):
        return self.train_data

    @property
    def num_batches(self):
        return self.file_paths_train.shape[0] // self.batch_size

    @property
    def num_batches_train(self):
        return self.num_batches

    @property
    def num_batches_valid(self):
        return self.file_paths_valid.shape[0] // self.batch_size

    @property
    def num_batches_test(self):
        return self.file_paths_test.shape[0] // self.batch_size


class AbstractBatch(list):
    """
    Abstract mini-batch list, contains a list of samples and allows for access to sample attributes in array form

    :param file_paths: list - file paths which samples will use to load the data
    """
    
    def __init__(self, file_paths):
        super(AbstractBatch, self).__init__(file_paths)

    # get attr from the list of samples
    def __getattr__(self, attr):
        attr_vals = list(map(lambda file_path: file_path.__getattribute__(attr), self))

        # try to convert values into a numpy array
        try:
            attr_list = np.array(attr_vals)
        except:
            attr_list = attr_vals

        self.__setattr__(attr, attr_list)  # save value for future fetches
        return attr_list


# ----------------------- #
#          Demo           #
# ----------------------- #


class DemoProvider(AbstractProvider):
    def __init__(self, data_config, batch_size=50, shuffle_data=True):
        file_paths = range(data_config.get('num_samples', 100))
        super(DemoProvider, self).__init__(file_paths, data_config, batch_size, shuffle_data)

    def load_data(self, file_paths):
        return DemoBatch(file_paths, self.data_config)


class DemoBatch(AbstractBatch):
    def __init__(self, file_paths, data_config):
        super(AbstractBatch, self).__init__([DemoSample(file_path, data_config) for file_path in file_paths])


class DemoSample(object):
    funcs = [np.sin, np.cos, np.tan]

    def __init__(self, file_path, data_config):
        self.id = file_path % 3
        N = data_config.get('num', 100)

        self.x = np.linspace(0, 2*np.pi, N)
        self.y = self.funcs[self.id](self.x)



