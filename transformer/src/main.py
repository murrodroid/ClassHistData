import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from modules.data_import import import_data
from modules.utils import return_device, encode_labels
from modules.dataloaders import get_dataloaders
from modules.training import train_model
from modules.set_seed import set_seed

SEED = set_seed(42)

model_name = 'meta-llama//Llama-3.2-1B'
target = 'icd10h_category'

# hyperparams
hyperparams = {
    'learning_rate': 0.0004,
    'batch_size': 32,
    'num_epochs': 64,
    'dropout_rate': 0.55,
}

device = return_device()

df, _ = import_data(target=target)

texts = df["tidy_cod"].tolist()
labels, label2id, id2label = encode_labels(df[target].tolist())

train_dl, val_dl, test_dl = get_dataloaders(
    texts,
    labels,
    tokenizer_name=model_name,
    batch_size=hyperparams['batch_size'],
    max_length=512,
    seed=SEED,
)

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    classifier_dropout=hyperparams['dropout_rate'],
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config
).to(device)

history = train_model(
    model,
    train_dl,
    val_dl,
    device=device,
    num_epochs=hyperparams["num_epochs"],
    learning_rate=hyperparams["learning_rate"],
)