from abc import ABC, abstractmethod

from torch import nn
from omegaconf import DictConfig


class Task(ABC):
	@staticmethod
	def build_model(cfg: DictConfig, model_type: str) -> nn.Module:
		"""Build just the model (no data, optimizer, etc.) for parameter counting.
		Override in subclass to support max_parameters."""
		raise NotImplementedError(
			"build_model not implemented for this task. Cannot use max_parameters."
		)

	@abstractmethod
	def train(self, epoch: int) -> dict[str, float]:
		pass

	@abstractmethod
	def validate(self, epoch: int) -> dict[str, float]:
		"""Evaluate on validation set.

		Args:
			epoch: Current epoch number

		Returns:
			Dictionary of validation metrics
		"""
		pass

	def eval(self, epoch: int) -> dict[str, float]:
		"""Legacy method name for backward compatibility. Use validate() instead."""
		return self.validate(epoch)

	@abstractmethod
	def calculate_performance(self, metrics: dict[str, float]) -> float:
		pass

	def test(self, epoch: int = None) -> dict[str, float]:
		"""Evaluate on test set.

		Override this method if your task has a separate test set.
		If not implemented, returns None.

		Args:
			epoch: Current epoch number (optional, for logging purposes)

		Returns:
			Dictionary of test metrics, or None if no test set
		"""
		return None

	def get_parameter_count(self) -> int:
		return sum(param.numel() for param in self.model.parameters())

	def get_backbone_parameter_count(self) -> int:
		return sum(param.numel() for param in self.model.backbone.parameters())

	def perform_diagnostics(self):
		pass
