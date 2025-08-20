config = dict(
    learning_types = ['passive_random','passive_ordered','active'], # 'passive_random','passive_ordered','active'
    labeled_size = 0.1,
    test_size = 0.1,
    rounds = 10,
    committee_size = 3,
    query_batch_size = 10,
    token_types = [
        {'method': 'char', 'ngram': 3},
        {'method': 'word', 'ngram': 0}],
    target_col = 'tidy_cod',
    seed = 42,
    # training hyper-parameters
    dropout_rate = 0.5,
    epoches = 16,
    batch_size = 32,
)