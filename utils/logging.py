import json
import os
from omegaconf import OmegaConf


class LocalLogger:
	def __init__(self, cfg, task_name: str, model_name: str):
		self.metrics = {}  # Stores metrics as {'name': [(step_a, value_a), (step_b, value_b), ...]}
		
		wandb_cfg = cfg.logging.wandb
		self.use_wandb = wandb_cfg.get("enabled", False)
		
		if self.use_wandb:
			import wandb
			self.wandb = wandb
			timestamp = os.path.basename(os.getcwd())
			task_subtype = cfg.get("task", {}).get("subtype")
			if task_subtype is not None:
				task_name += f"_{task_subtype}"
			run_name = f"{task_name}_{model_name}_{timestamp}"
						
			wandb.init(
				project=wandb_cfg.get("project", "smctm"),
				entity=wandb_cfg.get("entity", None),
				config=OmegaConf.to_container(cfg, resolve=True),
				tags=wandb_cfg.get("tags", []),
				name=run_name,
			)

	def log(self, name: str, value: float, step: int):
		if name not in self.metrics:
			self.metrics[name] = []
		self.metrics[name].append((step, value))

		if self.use_wandb:
			self.wandb.log({name: value}, step=step)

	def get_metrics(self) -> dict:
		return self.metrics

	def save(self, filepath: str):
		with open(filepath, "w") as f:
			json.dump(self.metrics, f)
	
	def finish(self):
		"""Finish wandb run if active"""
		if self.use_wandb:
			self.wandb.finish()
