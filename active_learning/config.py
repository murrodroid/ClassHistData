config = dict(
    # gen. info
    learning_types = ['active_random','passive_random','active_ordered','passive_ordered'], 
    labeled_size = 0.10,
    test_size = 0.15,
    token_types = [
        {'method': 'char', 'ngram': 3},
        {'method': 'char', 'ngram': 6},
        # {'method': 'word', 'ngram': 0},
        ],
    top_k = 3,
    target_col = 'tidy_cod',
    label_col = 'icd10h_category',
    hash_dim = 1<<16,
    seed = 42,
    # active learning
    passive_committee = True,
    rounds = 150,
    committee_size = 3,
    query_batch_size = 6,
    # training hyper-parameters
    epochs = 16,
    batch_size = 2048,
    dropout_rate = 0.5,
    lr = 1.4005e-3,
    weight_decay = 1e-5,
    hidden = 512,
    num_blocks = 2,
    expansion = 2,
    input_log1p = True,
    # perf knobs
    mixed_precision = True,
    dataloader_workers = 0, 
    pin_memory = True,
    prefetch_factor = 2,
    # wandb integration
    wandb_mode = 'online',
	wandb_project = 'active-learning-conference',
)
