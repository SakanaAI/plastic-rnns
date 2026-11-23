"""Episodic Sequential Split-MNIST with independent label mappings for A and B.

Adapted from the continual-learning setup of Irie et al. (2025).
"""

import os
import random
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from PIL import Image

from models import construct_rnn
from tasks.task import Task
from tasks.utils import get_device
from models.backbones import FewShotImageClassificationBackbone


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SequentialSplitMNISTModel(nn.Module):

    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str):
        super().__init__()
        self.num_classes = task_cfg.get('out_dims', 5)

        self.backbone = FewShotImageClassificationBackbone(d_input=model_cfg.d_input, norm_image_features=True)

        self.rnn = construct_rnn(
            d_input=model_cfg.d_input,
            d_model=model_cfg.d_model,
            model_type=model_type,
            model_kwargs=model_cfg
        )
        self.output_proj = nn.Linear(model_cfg.d_model, self.num_classes)

    def forward(self, input, aux_input):
        """
        input: [T, B, C, H, W] images
        aux_input: [T, B] label indices (num_classes = no-label token for queries)
        """
        assert input.ndim == 5
        assert aux_input.ndim == 2

        T, B, C, H, W = input.shape
        input = input.view(T * B, C, H, W)

        label_embeddings = torch.nn.functional.one_hot(aux_input.view(-1), num_classes=self.num_classes + 1).float()
        embedding = self.backbone(input, label_embeddings)
        embedding = embedding.view(T, B, -1)

        hidden, _ = self.rnn(embedding, state=self.rnn.get_initial_state())
        return self.output_proj(hidden)


class SequentialSplitMNISTLoss(nn.Module):
    """Sum mean query losses for Set A learning, Set A retention, and Set B learning."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _avg_ce_loss(self, predictions, targets, indices):
        loss = 0
        for idx in indices:
            loss += nn.CrossEntropyLoss()(predictions[idx], targets[idx])
        return loss / len(indices)

    def forward(self, predictions, targets, sequence_info):
        losses = {}
        losses['immediate'] = self._avg_ce_loss(predictions, targets, sequence_info['immediate_query_indices'])
        losses['forward'] = self._avg_ce_loss(predictions, targets, sequence_info['forward_query_indices'])

        total_loss = losses['immediate'] + losses['forward']

        losses['backward'] = self._avg_ce_loss(predictions, targets, sequence_info['backward_query_indices'])
        total_loss = total_loss + losses['backward']

        return total_loss, {'losses': losses}


class SequentialSplitMNISTDataset:
    """Disjoint MNIST support/query pools for digits 0–4 (A) and 5–9 (B)."""

    def __init__(self, phase: str = 'train', dataset_mode: str = 'split_mnist'):
        if phase not in {'train', 'val', 'test'}:
            raise ValueError(f"Unknown dataset phase: {phase}")
        if dataset_mode != 'split_mnist':
            raise ValueError("SequentialSplitMNIST only supports dataset_mode='split_mnist'")
        self.phase = phase
        self.n_classes = 5
        self.set_a_classes = [0, 1, 2, 3, 4]
        self.set_b_classes = [5, 6, 7, 8, 9]
        self.image_size = 32

        data_dir = os.path.join(_PROJECT_ROOT, 'data', 'mnist')
        mnist = torchvision.datasets.MNIST(root=data_dir, train=phase != 'test', download=True)
        # Split support and query pools before sampling episodes to avoid image leakage.
        support_slice, query_slice = {
            'train': (slice(0, 40000), slice(40000, 50000)),
            'val': (slice(50000, 55000), slice(55000, 60000)),
            'test': (slice(0, 5000), slice(5000, None)),
        }[phase]
        self.exemplar_data = mnist.data[support_slice].numpy()
        self.exemplar_labels = mnist.targets[support_slice].numpy()
        self.query_data = mnist.data[query_slice].numpy()
        self.query_labels = mnist.targets[query_slice].numpy()
        self.exemplar_label2inds = self._build_label_index(self.exemplar_labels)
        self.query_label2inds = self._build_label_index(self.query_labels)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    @staticmethod
    def _build_label_index(labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            label2inds.setdefault(int(label), []).append(idx)
        return label2inds

    def sample_exemplars(self, classes: list, n_per_class: int) -> list:
        examples = []
        for cls in classes:
            indices = self.exemplar_label2inds[cls]
            sampled = random.sample(indices, min(n_per_class, len(indices)))
            examples.extend((idx, cls, True) for idx in sampled)
        return examples

    def sample_query(self, classes: list) -> tuple:
        cls = random.choice(classes)
        idx = random.choice(self.query_label2inds[cls])
        return idx, cls, False

    def get_image(self, idx: int, is_exemplar: bool) -> torch.Tensor:
        data = self.exemplar_data if is_exemplar else self.query_data
        return self.transform(Image.fromarray(data[idx], mode='L'))


class SequentialSplitMNISTDataloader:

    def __init__(self,
                 dataset: SequentialSplitMNISTDataset,
                 internal_ticks: int = 1,
                 n_exemplars_per_class: int = 1,
                 queries_per_loss: int = 5,
                 batch_size: int = 64,
                 epoch_size: int = 10000):
        self.dataset = dataset
        self.internal_ticks = internal_ticks
        self.n_exemplars_per_class = n_exemplars_per_class
        self.queries_per_loss = queries_per_loss
        self.n_classes = dataset.n_classes
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.n_exemplars_total = self.n_classes * self.n_exemplars_per_class

        # Episode: A support, A query, B support, A retention, B query.
        task_a_end = self.n_exemplars_total
        task_b_end = task_a_end + self.queries_per_loss + self.n_exemplars_total
        self.total_images = task_b_end + self.queries_per_loss + self.queries_per_loss

        # Tick-expanded query indices (last tick of each query image)
        immediate_start = task_a_end
        backward_start = task_b_end
        forward_start = task_b_end + self.queries_per_loss

        self.sequence_info = {
            'immediate_query_indices': [(immediate_start + i + 1) * internal_ticks - 1
                                       for i in range(self.queries_per_loss)],
            'backward_query_indices': [(backward_start + i + 1) * internal_ticks - 1
                                      for i in range(self.queries_per_loss)],
            'forward_query_indices': [(forward_start + i + 1) * internal_ticks - 1
                                     for i in range(self.queries_per_loss)],
        }

        self.image_size = dataset.image_size

    def sample_episode(self):
        # Each domain gets an independent class-to-label permutation every episode.
        task_a_classes = self.dataset.set_a_classes.copy()
        random.shuffle(task_a_classes)
        task_a_label_map = {orig: new for new, orig in enumerate(task_a_classes)}

        task_b_classes = self.dataset.set_b_classes.copy()
        random.shuffle(task_b_classes)
        task_b_label_map = {orig: new for new, orig in enumerate(task_b_classes)}

        task_a_exemplars = self.dataset.sample_exemplars(self.dataset.set_a_classes, self.n_exemplars_per_class)
        random.shuffle(task_a_exemplars)
        task_b_exemplars = self.dataset.sample_exemplars(self.dataset.set_b_classes, self.n_exemplars_per_class)
        random.shuffle(task_b_exemplars)

        if self.queries_per_loss == self.n_classes:
            # One query per class
            queries_a_imm = [self.dataset.sample_query([cls]) for cls in self.dataset.set_a_classes]
            queries_a_bwd = [self.dataset.sample_query([cls]) for cls in self.dataset.set_a_classes]
            queries_b_fwd = [self.dataset.sample_query([cls]) for cls in self.dataset.set_b_classes]
        else:
            queries_a_imm = [self.dataset.sample_query(self.dataset.set_a_classes) for _ in range(self.queries_per_loss)]
            queries_a_bwd = [self.dataset.sample_query(self.dataset.set_a_classes) for _ in range(self.queries_per_loss)]
            queries_b_fwd = [self.dataset.sample_query(self.dataset.set_b_classes) for _ in range(self.queries_per_loss)]

        sequence = []
        for item in task_a_exemplars:
            sequence.append((*item, task_a_label_map))
        for query in queries_a_imm:
            sequence.append((*query, task_a_label_map))
        for item in task_b_exemplars:
            sequence.append((*item, task_b_label_map))
        for query in queries_a_bwd:
            sequence.append((*query, task_a_label_map))
        for query in queries_b_fwd:
            sequence.append((*query, task_b_label_map))

        images, aux_inputs, targets = [], [], []
        for img_idx, orig_label, is_exemplar, label_map in sequence:
            img = self.dataset.get_image(img_idx, is_exemplar)
            remapped_label = label_map[orig_label]
            for _ in range(self.internal_ticks):
                images.append(img)
                aux_inputs.append(remapped_label if is_exemplar else self.n_classes)
                targets.append(remapped_label)

        return (torch.stack(images), torch.tensor(aux_inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long), self.sequence_info)

    def __iter__(self):
        for _ in range(len(self)):
            batch_inputs, batch_aux, batch_targets = [], [], []
            for _ in range(self.batch_size):
                inputs, aux_inputs, targets, _ = self.sample_episode()
                batch_inputs.append(inputs)
                batch_aux.append(aux_inputs)
                batch_targets.append(targets)

            inputs = torch.stack(batch_inputs)    # [B, T, 3, H, W]
            aux_inputs = torch.stack(batch_aux)   # [B, T]
            targets = torch.stack(batch_targets)  # [B, T]
            yield (inputs, aux_inputs), targets

    def __len__(self):
        return self.epoch_size // self.batch_size


class SequentialSplitMNISTTask(Task):
    @staticmethod
    def build_model(cfg, model_type):
        model = SequentialSplitMNISTModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type)
        with torch.no_grad():
            model(torch.zeros(1, 1, 3, 32, 32), aux_input=torch.zeros(1, 1, dtype=torch.long))
        return model

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = get_device(cfg)

        task_cfg = cfg.task
        self.dataset_mode = task_cfg.get('dataset_mode', 'split_mnist')
        self.n_exemplars_per_class = task_cfg.get('n_exemplars_per_class', 1)
        self.internal_ticks = task_cfg.get('iterations', 1)
        self.queries_per_loss = task_cfg.get('queries_per_loss', 5)

        train_dataset = SequentialSplitMNISTDataset(phase='train', dataset_mode=self.dataset_mode)
        val_dataset = SequentialSplitMNISTDataset(phase='val', dataset_mode=self.dataset_mode)
        test_dataset = SequentialSplitMNISTDataset(phase='test', dataset_mode=self.dataset_mode)

        loader_kwargs = dict(
            internal_ticks=self.internal_ticks,
            n_exemplars_per_class=self.n_exemplars_per_class,
            queries_per_loss=self.queries_per_loss,
            batch_size=cfg.training.batch_size,
        )
        self.train_dataloader = SequentialSplitMNISTDataloader(
            dataset=train_dataset, epoch_size=cfg.training.batch_size * 100, **loader_kwargs
        )
        self.val_dataloader = SequentialSplitMNISTDataloader(
            dataset=val_dataset, epoch_size=cfg.training.batch_size * 20, **loader_kwargs
        )
        self.test_dataloader = SequentialSplitMNISTDataloader(
            dataset=test_dataset, epoch_size=cfg.training.batch_size * 20, **loader_kwargs
        )

        self.sequence_info = self.train_dataloader.sequence_info
        self.n_classes = train_dataset.n_classes

        self.model = SequentialSplitMNISTModel(
            model_cfg=cfg.model, task_cfg=task_cfg,
            model_type=HydraConfig.get().runtime.choices.model
        ).to(self.device)
        self._init_lazy_modules()

        self.loss_fn = SequentialSplitMNISTLoss(n_classes=self.n_classes)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        self._setup_scheduler(cfg)
        self.max_grad_norm = cfg.training.max_grad_norm

        print(f"Dataset mode: {self.dataset_mode}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Queries per loss: {self.queries_per_loss}")
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

    def _init_lazy_modules(self):
        T = self.train_dataloader.total_images * self.internal_ticks
        dummy_input = torch.zeros(T, 1, 3, 32, 32, device=self.device)
        dummy_aux = torch.zeros(T, 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model(dummy_input, aux_input=dummy_aux)

    def _setup_scheduler(self, cfg):
        sched = cfg.training.learn_rate_scheduler
        if sched == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.training.epochs,
                eta_min=cfg.training.learning_rate * cfg.training.learn_rate_scheduler_final_factor
            )
        elif sched == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1,
                end_factor=cfg.training.learn_rate_scheduler_final_factor,
                total_iters=cfg.training.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )

    def _compute_accuracy(self, logits, targets, indices):
        total_acc = 0
        for idx in indices:
            pred = logits[idx].argmax(dim=1)
            total_acc += (pred == targets[idx]).float().mean().item()
        return total_acc / len(indices)

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0
        total_immediate_acc = 0
        total_forward_acc = 0
        total_backward_acc = 0
        total_grad_norm = 0
        n_batches = 0

        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for (inputs, aux_inputs), targets in pbar:
            inputs = inputs.transpose(0, 1).contiguous().to(self.device)
            aux_inputs = aux_inputs.transpose(0, 1).contiguous().to(self.device)
            targets = targets.transpose(0, 1).contiguous().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs, aux_input=aux_inputs)
            loss, info = self.loss_fn(logits, targets, self.sequence_info)
            loss.backward()

            grad_norm = torch.sqrt(sum(
                p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None
            ))
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            immediate_acc = self._compute_accuracy(logits, targets, self.sequence_info['immediate_query_indices'])
            forward_acc = self._compute_accuracy(logits, targets, self.sequence_info['forward_query_indices'])
            backward_acc = self._compute_accuracy(logits, targets, self.sequence_info['backward_query_indices'])

            total_loss += loss.item()
            total_immediate_acc += immediate_acc
            total_forward_acc += forward_acc
            total_backward_acc += backward_acc
            total_grad_norm += grad_norm.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "imm": f"{immediate_acc:.3f}",
                "fwd": f"{forward_acc:.3f}",
                "bwd": f"{backward_acc:.3f}"
            })

        self.scheduler.step()

        metrics = {
            "loss": total_loss / n_batches,
            "grad_norm": total_grad_norm / n_batches,
            "accuracy_immediate": total_immediate_acc / n_batches,
            "accuracy_forward": total_forward_acc / n_batches,
            "accuracy_backward": total_backward_acc / n_batches,
            "accuracy": (total_immediate_acc + total_forward_acc + total_backward_acc) / (3 * n_batches),
            "lr": self.scheduler.get_last_lr()[0]
        }

        synapses = getattr(getattr(self.model.rnn, 'recurrent_module', None), 'synapses', None)
        if synapses is not None and hasattr(synapses, 'beta'):
            metrics["plastic_beta"] = synapses.beta.item()

        return metrics

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_immediate_acc = 0
        total_forward_acc = 0
        total_backward_acc = 0
        n_batches = 0

        for (inputs, aux_inputs), targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                inputs = inputs.transpose(0, 1).contiguous().to(self.device)
                aux_inputs = aux_inputs.transpose(0, 1).contiguous().to(self.device)
                targets = targets.transpose(0, 1).contiguous().to(self.device)

                logits = self.model(inputs, aux_input=aux_inputs)
                loss, info = self.loss_fn(logits, targets, self.sequence_info)

                total_loss += loss.item()
                total_immediate_acc += self._compute_accuracy(logits, targets, self.sequence_info['immediate_query_indices'])
                total_forward_acc += self._compute_accuracy(logits, targets, self.sequence_info['forward_query_indices'])
                total_backward_acc += self._compute_accuracy(logits, targets, self.sequence_info['backward_query_indices'])
                n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "accuracy_immediate": total_immediate_acc / n_batches,
            "accuracy_forward": total_forward_acc / n_batches,
            "accuracy_backward": total_backward_acc / n_batches,
            "accuracy": (total_immediate_acc + total_forward_acc + total_backward_acc) / (3 * n_batches),
        }

    def validate(self, epoch: int) -> dict[str, float]:
        return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
        return self._evaluate(self.test_dataloader, desc)

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
        acc_keys = ["eval_accuracy_immediate", "eval_accuracy_forward", "eval_accuracy_backward"]
        total = 0.0
        count = 0
        for key in acc_keys:
            if key in metrics:
                values = [pair[1] for pair in metrics[key]]
                recent = values[-window_size:]
                if recent:
                    total += sum(recent) / len(recent)
                    count += 1
        if count > 0:
            return total
        if "eval_accuracy" in metrics:
            accuracies = [pair[1] for pair in metrics["eval_accuracy"]]
            recent = accuracies[-window_size:]
            return sum(recent) / len(recent) if recent else 0.0
        return 0.0
