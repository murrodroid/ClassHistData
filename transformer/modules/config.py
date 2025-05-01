# modules/config.py
hyperparams = dict(
    learning_rate=4e-4,
    batch_size   =32,
    num_epochs   =64,
    dropout_rate =0.55,
    max_length   =512,
)
model_name = "meta-llama/Llama-3.2-1B"
target     = "icd10h_category"
