from .config import config
from .import_data import import_data
from .utils import update_indexes, labeled_unlabeled_test_split, create_tokenized_df
from .text_preprocess import tokenize
from .networks import network_qpidgram

df,_ = import_data()

df = create_tokenized_df(df)

labeled_idx, unlabeled_idx, test_idx = labeled_unlabeled_test_split(df)

data_idx = dict(
    labeled = labeled_idx,
    unlabeled = unlabeled_idx,
    test = test_idx
)

def train_model(data_idx,network,config=config):
    # create tensors for data
    pass
    # train model
    pass

def train_committee(config=config):
    # randomize weights but train models on same arch
    pass

def test_model(data_idx, config=config):
    pass

def train_passive(data_idx, ordered=False, config=config):
    test_accuracies = []
    
    for round in range(config['rounds']):
        # train model
        model = train_model(data_idx)
        # check test acc & save to test_accuracies
        test_accuracies += test_model(model)
        # add labels to labeled pool
        ## create func which takes data_idx and changes it to match
        data_idx = update_indexes(data_idx,active=False,ordered=False)
    
    return test_accuracies

def train_active():
    pass

def train(data_idx,config=config):
    learning_types = config['learning_types']
    if 'passive_random' in learning_types:
        pass
    if 'passive_ordered' in learning_types:
        pass
    if 'active' in learning_types:
        pass

def train_all_types():
    pass