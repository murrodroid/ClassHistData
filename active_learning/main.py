# imports
from .config import config
from .import_data import import_data
from .utils import labeled_unlabeled_test_split
from .training import train
# dataset
df,_ = import_data()

labeled_idx, unlabeled_idx, test_idx = labeled_unlabeled_test_split(df, labeled_size=config['labeled_size'])

data_idx = dict(
    labeled = labeled_idx,
    unlabeled = unlabeled_idx,
    test = test_idx
)

# overhead func which decides which types of loops to run (passive-random, passive-ordered, active)
for learning_type in config['learning_types']:
    train(learning_type=learning_type)
    

    # run loop for each type and save data


# plot and save
