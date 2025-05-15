from __future__ import annotations
from pathlib import Path
import wandb

class WandBLogger:
    """
    Tiny façade so the rest of the code never imports wandb directly.
    Turn it off by passing logger=None.
    """
    def __init__(self, config, run_dir: Path):
        self.run = wandb.init(
            project=config["project"],
            entity =config.get("entity"),
            mode   =config["mode"],      # respects OFFLINE on HPC nodes
            dir    =str(run_dir),
            config =config.get("hyperparams"),  # log HPs
            name   =run_dir.name,
            reinit =True
        )

    def log(self, metrics: dict, step: int | None = None):
        self.run.log(metrics, step=step)

    
    def log_artifact(self, path, *, name: str, type_: str = "model"):

        art = wandb.Artifact(name=name, type=type_)
        path = Path(path)

        if path.is_dir():
            art.add_dir(str(path))                  # folder → add_dir ✅
        else:
            art.add_file(str(path))                 # single file → add_file ✅

        self.run.log_artifact(art)

    def finish(self):
        self.run.finish()
