"""Gradient-based and metric-based few-shot learning baselines.

The models in this module consume the same sequence representation as the
recurrent few-shot models: labelled support images come first and have a
one-hot ``aux_input``; unlabelled query images have an all-zero ``aux_input``.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call


class Conv4Encoder(nn.Module):
    """The four-block convolutional encoder used by MAML and ProtoNet."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        track_running_stats: bool,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        blocks = []
        in_channels = 3
        for _ in range(4):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(
                        hidden_channels,
                        track_running_stats=track_running_stats,
                    ),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_channels = hidden_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs).flatten(start_dim=1)


def _validate_episode_inputs(
    inputs: torch.Tensor,
    aux_input: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if inputs.ndim != 5:
        raise ValueError("inputs must have shape T x B x C x H x W")
    if aux_input.ndim != 3:
        raise ValueError("aux_input must have shape T x B x num_classes")
    if inputs.shape[:2] != aux_input.shape[:2]:
        raise ValueError("inputs and aux_input must share T and B dimensions")
    if aux_input.shape[-1] != num_classes:
        raise ValueError(
            f"expected {num_classes} auxiliary label dimensions, "
            f"got {aux_input.shape[-1]}"
        )

    support_mask = aux_input.sum(dim=-1) > 0
    if not torch.all(support_mask.any(dim=0)):
        raise ValueError("every episode must contain at least one support image")
    if not torch.all((~support_mask).any(dim=0)):
        raise ValueError("every episode must contain at least one query image")
    return support_mask


class ProtoNet(nn.Module):
    """Prototypical Network with squared-Euclidean query logits."""

    requires_inner_grad = False

    def __init__(
        self,
        *,
        num_classes: int,
        hidden_channels: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_classes = num_classes
        self.temperature = temperature
        self.backbone = Conv4Encoder(
            hidden_channels,
            track_running_stats=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        aux_input: torch.Tensor,
    ) -> torch.Tensor:
        support_mask = _validate_episode_inputs(
            inputs,
            aux_input,
            self.num_classes,
        )
        time_steps, batch_size, channels, height, width = inputs.shape
        embeddings = self.backbone(
            inputs.reshape(time_steps * batch_size, channels, height, width)
        ).reshape(time_steps, batch_size, -1)

        episode_logits = []
        for batch_index in range(batch_size):
            episode_support_mask = support_mask[:, batch_index]
            support_embeddings = embeddings[
                episode_support_mask,
                batch_index,
            ]
            support_labels = aux_input[
                episode_support_mask,
                batch_index,
            ].argmax(dim=-1)
            support_one_hot = F.one_hot(
                support_labels,
                num_classes=self.num_classes,
            ).to(support_embeddings.dtype)
            class_counts = support_one_hot.sum(dim=0)
            if torch.any(class_counts == 0):
                raise ValueError("every class must have at least one support image")

            prototypes = (
                support_one_hot.transpose(0, 1) @ support_embeddings
            ) / class_counts.unsqueeze(-1)
            squared_distances = torch.cdist(
                embeddings[:, batch_index],
                prototypes,
                p=2,
            ).square()
            episode_logits.append(-squared_distances / self.temperature)

        return torch.stack(episode_logits, dim=1)


class _MAMLLearner(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden_channels: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.encoder = Conv4Encoder(
            hidden_channels,
            # Per-batch statistics match the episode-local normalization used
            # by the original MAML implementation and avoid shared task state.
            track_running_stats=False,
        )
        with torch.no_grad():
            feature_size = self.encoder(
                torch.zeros(2, 3, image_size, image_size)
            ).shape[-1]
        self.classifier = nn.Linear(feature_size, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The reference implementation uses Xavier initialization for both the
        # convolutions and the final linear classifier.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class MAML(nn.Module):
    """Full MAML adaptation of a Conv4 encoder and linear classifier."""

    requires_inner_grad = True

    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
        hidden_channels: int = 32,
        inner_learning_rate: float = 0.01,
        inner_steps: int = 5,
        first_order: bool = False,
    ) -> None:
        super().__init__()
        if inner_learning_rate <= 0:
            raise ValueError("inner_learning_rate must be positive")
        if inner_steps < 1:
            raise ValueError("inner_steps must be at least one")

        self.num_classes = num_classes
        self.inner_learning_rate = inner_learning_rate
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.learner = _MAMLLearner(
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            image_size=image_size,
        )

    @property
    def backbone(self) -> nn.Module:
        return self.learner.encoder

    def _adapt_episode(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
        query_inputs: torch.Tensor,
    ) -> torch.Tensor:
        parameters = OrderedDict(self.learner.named_parameters())
        create_graph = self.training and not self.first_order

        # Evaluation still needs gradients for task adaptation. The returned
        # query logits are detached below when no outer update will be taken.
        with torch.enable_grad():
            for _ in range(self.inner_steps):
                support_logits = functional_call(
                    self.learner,
                    parameters,
                    (support_inputs,),
                )
                # Finn et al. average over shots but sum over ways. This keeps
                # the reference inner step size (0.01) on the same scale.
                shots_per_class = support_targets.numel() / self.num_classes
                support_loss = F.cross_entropy(
                    support_logits,
                    support_targets,
                    reduction="sum",
                ) / shots_per_class
                gradients = torch.autograd.grad(
                    support_loss,
                    tuple(parameters.values()),
                    create_graph=create_graph,
                )
                parameters = OrderedDict(
                    (
                        name,
                        parameter - self.inner_learning_rate * gradient,
                    )
                    for (name, parameter), gradient in zip(
                        parameters.items(),
                        gradients,
                        strict=True,
                    )
                )

            query_logits = functional_call(
                self.learner,
                parameters,
                (query_inputs,),
            )

        return query_logits if self.training else query_logits.detach()

    def forward(
        self,
        inputs: torch.Tensor,
        aux_input: torch.Tensor,
    ) -> torch.Tensor:
        support_mask = _validate_episode_inputs(
            inputs,
            aux_input,
            self.num_classes,
        )
        time_steps, batch_size = inputs.shape[:2]
        episode_logits = []

        for batch_index in range(batch_size):
            episode_support_mask = support_mask[:, batch_index]
            episode_query_mask = ~episode_support_mask
            support_targets = aux_input[
                episode_support_mask,
                batch_index,
            ].argmax(dim=-1)
            if torch.unique(support_targets).numel() != self.num_classes:
                raise ValueError("every class must have at least one support image")

            query_logits = self._adapt_episode(
                inputs[episode_support_mask, batch_index],
                support_targets,
                inputs[episode_query_mask, batch_index],
            )
            logits = query_logits.new_zeros(time_steps, self.num_classes)
            logits[episode_query_mask] = query_logits
            episode_logits.append(logits)

        return torch.stack(episode_logits, dim=1)
