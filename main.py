import json
import logging
import os
import timeit

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

from tasks import construct_task, _TASKS
from tasks.utils import optimise_model_size
from utils import LocalLogger, plot_metrics, set_random_seeds, save_checkpoint, load_checkpoint, get_rng_state, set_rng_state, find_latest_checkpoint

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

def log_metrics(logger, metrics: dict[str, float], step: int, prefix: str = ""):
	for name, value in metrics.items():
		logger.log(f"{prefix}_{name}", value, step)

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> float:

	if cfg.checkpoint.resume_from is not None:
		hydra_cfg = HydraConfig.get()
		original_cwd = hydra_cfg.runtime.cwd

		resume_path = cfg.checkpoint.resume_from
		if not os.path.isabs(resume_path):
			resume_path = os.path.join(original_cwd, resume_path)

		checkpoint_path = find_latest_checkpoint(resume_path)
		checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
		if os.getcwd() != checkpoint_dir:
			os.chdir(checkpoint_dir)
			log.info(f"Changed to checkpoint directory: {checkpoint_dir}")
		cfg.checkpoint.resume_from = checkpoint_path

	log.info(OmegaConf.to_yaml(cfg))
	set_random_seeds(cfg.seed)

	task_type = HydraConfig.get().runtime.choices.task
	model_type = HydraConfig.get().runtime.choices.model

	summary = {}

	np.random.seed(cfg.seed)

	if "max_parameters" in cfg:
		cfg = optimise_model_size(_TASKS[task_type], cfg, model_type)

	logger = LocalLogger(cfg=cfg, task_name=task_type, model_name=model_type)

	task = construct_task(cfg, task_type=task_type)

	if cfg.training.compile:
		log.info("Compiling model with torch.compile(mode='reduce-overhead')...")
		task.model = torch.compile(task.model, mode="reduce-overhead")

	start_epoch = 1
	if cfg.checkpoint.resume_from is not None:
		checkpoint = load_checkpoint(cfg.checkpoint.resume_from)
		if checkpoint is not None:
			task.model.load_state_dict(checkpoint["model_state_dict"])
			task.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			task.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
			set_rng_state(checkpoint["rng_state"])
			start_epoch = checkpoint["epoch"] + 1
			logger.metrics = checkpoint["metrics"]
			log.info(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")

	timer = timeit.default_timer()
	pbar = tqdm(range(start_epoch, cfg.training.epochs + 1), desc="Training", initial=start_epoch - 1, total=cfg.training.epochs)
	for epoch in pbar:

		train_metrics = task.train(epoch)

		log_metrics(logger, train_metrics, epoch, prefix="train")

		if epoch % cfg.evaluation.evaluation_interval == 0 or epoch == 1:
			eval_metrics = task.eval(epoch)
			log_metrics(logger, eval_metrics, epoch, prefix="eval")

			test_metrics = task.test(epoch)
			if test_metrics is not None:
				log_metrics(logger, test_metrics, epoch, prefix="test")

			if epoch % cfg.logging.plot_interval == 0:
				plot_metrics(logger.get_metrics())

			pbar_desc = f"Val: {eval_metrics}"
			if test_metrics is not None:
				pbar_desc += f" | Test: {test_metrics}"
			pbar.set_description(pbar_desc)

		if cfg.checkpoint.enabled and epoch % cfg.checkpoint.checkpoint_every == 0:
			checkpoint_path = save_checkpoint(
				epoch=epoch,
				model_state=task.model.state_dict(),
				optimizer_state=task.optimizer.state_dict(),
				scheduler_state=task.scheduler.state_dict(),
				metrics=logger.get_metrics(),
				rng_state=get_rng_state(),
			)
			log.info(f"Saved checkpoint to {checkpoint_path}")

	logger.save("metrics.json")

	summary["parameter_count"] = task.get_parameter_count()
	summary["runtime"] = timeit.default_timer() - timer
	summary["performance"] = task.calculate_performance(logger.get_metrics())

	if "test_accuracy" in logger.get_metrics():
		final_test_metrics = {
			name.replace("test_", ""): values[-1][1]
			for name, values in logger.get_metrics().items()
			if name.startswith("test_")
		}
		summary["test_metrics"] = final_test_metrics
		log.info(f"Final test metrics: {final_test_metrics}")

	with open("summary.json", "w") as f:
		json.dump(summary, f)

	task.perform_diagnostics()  # Perform final task/model diagnostics
	log.info(f"Performance: {summary['performance']}")

	logger.finish()

	return summary["performance"]


if __name__ == "__main__":
	main()
