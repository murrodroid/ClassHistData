import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from modules.config        import (
    hyperparams,              
    model_name,              
    target,                   
    run_dir, ckpt_dir, reports_dir,
    wandb_cfg,
)

from modules.set_seed       import set_seed
from modules.utils          import return_device, encode_labels
from modules.save_utils     import save_history
from modules.data_import    import import_data
from modules.dataloaders    import get_dataloaders
from modules.training       import train_model
from modules.eval           import evaluate
from modules.logger_wandb   import WandBLogger   

seed = set_seed(wandb_cfg.get("seed", 42))
print(f"▶ Using seed: {seed}")
print(f"▶ Using model: {model_name}")
print(f"▶ Using target: {target}")

device = return_device()
df, _ = import_data(target=target)

texts  = df["tidy_cod"].tolist()
labels_int, label2id, id2label = encode_labels(df[target].tolist())

train_dl, val_dl, test_dl, tokenizer = get_dataloaders(
    texts,
    labels_int,
    tokenizer_name=model_name,
    batch_size=hyperparams["batch_size"],
    max_length=hyperparams["max_length"],
    seed=seed,
)
print(f'Dataloaders initialized: {len(train_dl)} train, {len(val_dl)} val, {len(test_dl)} test')

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    classifier_dropout=hyperparams["dropout_rate"],
    pad_token_id=tokenizer.pad_token_id,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config
).to(device)

logger = WandBLogger({**wandb_cfg, "hyperparams": hyperparams}, run_dir)

print('Training model...')
history = train_model(
    model,
    train_dl,
    val_dl,
    device=device,
    num_epochs=hyperparams["num_epochs"],
    learning_rate=hyperparams["learning_rate"],
    checkpoint_dir=ckpt_dir,
    tokenizer=tokenizer,    
    logger=logger,
    top_k=hyperparams["top_k"],              
)
print('Training complete!')

save_history(history, run_dir / "history.json")

if logger:
    logger.log_artifact(run_dir / "history.json", name="history", type_="metrics")

best_state = torch.load(ckpt_dir / "best.pt", map_location=device)
model.load_state_dict(best_state["model_state"])

report, cm, acc1, acck, f1, top1_pred, topk_pred = evaluate(
    model,
    test_dl,
    device,
    id2label,
    reports_dir=reports_dir,
    logger=logger,
    top_k=hyperparams["top_k"],
)

print(f"Top-1 accuracy  : {acc1:.3f}")
print(f"Top-{hyperparams['top_k']} accuracy: {acck:.3f}")
print(f"Weighted-F1     : {f1:.3f}\n")
print("===== Test-set classification report =====\n")
print(report)

if logger:
    logger.log({"run_dir": str(run_dir)})
    logger.finish()