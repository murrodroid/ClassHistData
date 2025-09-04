config = dict(
    # gen. info
    learning_types = ['passive_random','passive_ordered','active_random','active_ordered'], 
    labeled_size = 0.1,
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
    rounds = 50,
    committee_size = 3,
    query_batch_size = 5,
    # training hyper-parameters
    epochs = 16,
    batch_size = 512,
    dropout_rate = 0.5,
    lr = 1.3005e-3,
    weight_decay = 1e-5,
    hidden = 512,
    num_blocks = 2,
    expansion = 2,
    input_log1p = True,
)