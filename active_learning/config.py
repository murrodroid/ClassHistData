config = dict(
    # gen. info
    learning_types = ['passive_random','passive_ordered','active_random','active_ordered'], 
    labeled_size = 0.05,
    test_size = 0.2,
    token_types = [
        {'method': 'char', 'ngram': 3},
        {'method': 'char', 'ngram': 5},
        # {'method': 'word', 'ngram': 0},
        ],
    top_k = 3,
    target_col = 'tidy_cod',
    label_col = 'icd10h_category',
    hash_dim = 1<<16,
    seed = 333,
    # active learning
    passive_committee = True,
    rounds = 400,
    committee_size = 7,
    query_batch_size = 3,
    # training hyper-parameters
    epochs = 16,
    batch_size = 4096,
    dropout_rate = 0.5,
    lr = 1.3005e-3,
    weight_decay = 1e-5,
    hidden = 512,
    num_blocks = 2,
    expansion = 2,
    input_log1p = True,
    # perf knobs
    mixed_precision = True,
    dataloader_workers = 4, 
    pin_memory = True,
    prefetch_factor = 2,
)
