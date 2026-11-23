import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from abc import abstractmethod
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from models import construct_rnn
from models.backbones import SpatialTokenBackbone
from tasks.task import Task
from tasks.utils import get_device

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ImageClassificationModel(nn.Module):
   """Wrapper model for image classification that handles backbone, RNN, and output projection."""

   def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str, num_channels: int, img_size: int):
       super().__init__()
       self.backbone = nn.Flatten()
       flattened_size = num_channels * img_size * img_size
       self.input_proj = nn.Linear(flattened_size, model_cfg.d_input)
       self.rnn = construct_rnn(d_input=model_cfg.d_input, d_model=model_cfg.d_model, model_type=model_type, model_kwargs=model_cfg)
       self.output_proj = nn.Linear(model_cfg.d_model, task_cfg.out_dims)

   def forward(self, input, states):
       assert input.ndim == 5, "Input should be of size B x iterations x C x H x W"

       B, iterations, C, H, W = input.shape

       input_flat = input.view(B * iterations, C, H, W)
       embeddings = self.backbone(input_flat)  # [B*iterations, C*H*W]
       embeddings = self.input_proj(embeddings)  # [B*iterations, d_input]
       embeddings = embeddings.view(B, iterations, -1)  # [B, iterations, d_input]
       embeddings = embeddings.transpose(0, 1)
       hidden, final_state = self.rnn(embeddings, state=states)  # [iterations, B, d_model]
       logits = self.output_proj(hidden)  # [iterations, B, num_classes]
       logits = logits.permute(1, 2, 0)

       return logits, final_state

   def get_initial_states(self, batch_size):
       """Get initial RNN state. Batch size is not used since state is independent of batch."""
       return self.rnn.get_initial_state()

class CrossAttentionImageClassificationModel(nn.Module):
   """Model wrapper using a CNN backbone that produces spatial tokens for cross-attention CTM."""

   def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str, num_channels: int, img_size: int):
       super().__init__()
       backbone_dim = task_cfg.backbone_dim
       backbone_layers = task_cfg.backbone_layers
       self.backbone = SpatialTokenBackbone(
           in_channels=num_channels,
           hidden_dim=backbone_dim,
           num_blocks=backbone_layers,
       )
       spatial_size = img_size // (2 ** backbone_layers)
       num_tokens = spatial_size ** 2
       d_input = num_tokens * backbone_dim
       self.rnn = construct_rnn(d_input=d_input, d_model=model_cfg.d_model, model_type=model_type, model_kwargs=model_cfg)
       self.output_proj = nn.Linear(model_cfg.d_model, task_cfg.out_dims)

   def forward(self, input, states):
       assert input.ndim == 5, "Input should be of size B x iterations x C x H x W"
       B, iterations, C, H, W = input.shape
       input_flat = input.view(B * iterations, C, H, W)
       tokens = self.backbone(input_flat)  # (B*iter, num_tokens, token_dim)
       flat_tokens = tokens.view(B * iterations, -1)  # (B*iter, num_tokens*token_dim)
       flat_tokens = flat_tokens.view(B, iterations, -1).transpose(0, 1)  # (iter, B, D)
       hidden, final_state = self.rnn(flat_tokens, state=states)
       logits = self.output_proj(hidden).permute(1, 2, 0)
       return logits, final_state

   def get_initial_states(self, batch_size):
       return self.rnn.get_initial_state()


class ImageClassificationDataset(torch.utils.data.Dataset):
   DATASET_CONFIG = None  # dict with 'mean', 'std', 'image_size', 'padding', 'num_channels', 'num_classes'
   def __init__(self, phase='train', iterations=1, do_not_use_random_transf=False):
       assert self.DATASET_CONFIG is not None, "Subclass must define DATASET_CONFIG"
       assert phase in ['train', 'val', 'test'], f"Phase {phase} not valid"
      
       self.phase = phase
       self.iterations = iterations
       self.config = self.DATASET_CONFIG
      
       # Subclass must populate these
       self.data = None
       self.labels = None
       self._load_data(phase)
      
       assert self.data is not None and self.labels is not None, "Subclass must set self.data and self.labels in _load_data()"
      
       self.num_classes = self.config['num_classes']
      
       mean_pix = self.config['mean']
       std_pix = self.config['std']
       self.normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

       if (phase in ['test', 'val']) or do_not_use_random_transf:
           self.transform = transforms.Compose([
               transforms.ToTensor(),
               self.normalize
           ])
       else:
           img_size = self.config['image_size']
           padding = self.config['padding']
           use_hflip = self.config.get('use_horizontal_flip', True)
           transform_list = [transforms.RandomCrop(img_size, padding=padding)]
           if use_hflip:
               transform_list.append(transforms.RandomHorizontalFlip())
           transform_list.extend([
               transforms.ToTensor(),
               self.normalize
           ])
           self.transform = transforms.Compose(transform_list)
  
   @abstractmethod
   def _load_data(self, phase: str):
       """Load data and labels. Must set self.data and self.labels."""
       pass
  
   def __getitem__(self, index):
       img, label = self.data[index], self.labels[index]
       if len(img.shape) == 2:
           img = Image.fromarray(img, mode='L')
       else:
           img = Image.fromarray(img)
          
       if self.transform is not None:
           img = self.transform(img)
      
       img = img.unsqueeze(0).repeat(self.iterations, 1, 1, 1)
       target = torch.full((self.iterations,), label, dtype=torch.long)
       return img, target
  
   def __len__(self):
       return len(self.data)

class ImageClassificationTask(Task):
   DATASET_CLASS = None

   @classmethod
   def build_model(cls, cfg, model_type):
       dc = cls.DATASET_CLASS.DATASET_CONFIG
       model = ImageClassificationModel(
           model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type,
           num_channels=dc['num_channels'], img_size=dc['image_size']
       )
       with torch.no_grad():
           dummy = torch.zeros(1, 1, dc['num_channels'], dc['image_size'], dc['image_size'])
           model(dummy, states=model.get_initial_states(batch_size=1))
       return model

   def __init__(self, cfg: DictConfig):
       super().__init__()
       self.device = get_device(cfg)
       self.iterations = cfg.task.iterations
      
       assert self.DATASET_CLASS is not None, "Subclass must define DATASET_CLASS"
      
       train_dataset = self.DATASET_CLASS(
           phase='train',
           iterations=cfg.task.iterations
       )
       val_dataset = self.DATASET_CLASS(
           phase='val',
           iterations=cfg.task.iterations
       )
       test_dataset = self.DATASET_CLASS(
           phase='test',
           iterations=cfg.task.iterations
       )

       self.num_classes = train_dataset.num_classes
       self.dataset_config = train_dataset.config

       self.train_dataloader = DataLoader(
           train_dataset,
           batch_size=cfg.training.batch_size,
           shuffle=True,
           num_workers=0
       )
       self.val_dataloader = DataLoader(
           val_dataset,
           batch_size=cfg.training.batch_size,
           shuffle=False,
           num_workers=0
       )
       self.test_dataloader = DataLoader(
           test_dataset,
           batch_size=cfg.training.batch_size,
           shuffle=False,
           num_workers=0
       )
      
       model_type = HydraConfig.get().runtime.choices.model
       num_channels = self.dataset_config['num_channels']
       img_size = self.dataset_config['image_size']
       self.model = self._create_model(cfg, model_type, num_channels, img_size).to(self.device)
       self.init_lazy_modules()
       print(f"Total Parameter Count: {self.get_parameter_count()}")
       print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")


       self.loss_fn = nn.CrossEntropyLoss()
       self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

       assert cfg.training.learn_rate_scheduler in ("none", "cosine_annealing", "linear", "cosine_annealing_warm_restarts", "multi_step")
       if cfg.training.learn_rate_scheduler == "none":
           self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
       elif cfg.training.learn_rate_scheduler == "cosine_annealing":
           self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.learning_rate * cfg.training.learn_rate_scheduler_final_factor)
       elif cfg.training.learn_rate_scheduler == "linear":
           self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=cfg.training.learn_rate_scheduler_final_factor, total_iters=cfg.training.epochs)
       elif cfg.training.learn_rate_scheduler == "cosine_annealing_warm_restarts":
           self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=cfg.training.learning_rate * cfg.training.learn_rate_scheduler_final_factor)
       elif cfg.training.learn_rate_scheduler == "multi_step":
           self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(cfg.training.epochs*0.5), int(cfg.training.epochs*0.75)], gamma=0.1)

       self.max_grad_norm = cfg.training.max_grad_norm

   def _create_model(self, cfg, model_type, num_channels, img_size):
       return ImageClassificationModel(
           model_cfg=cfg.model,
           task_cfg=cfg.task,
           model_type=model_type,
           num_channels=num_channels,
           img_size=img_size
       )

   def train(self, epoch: int) -> dict[str, float]:
       self.model.train()
       objectives = []
       pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
       for inputs, targets in pbar:
           inputs = inputs.to(self.device)  # [B, iterations, C, H, W]
           targets = targets.to(self.device)  # [B, iterations]
          
           self.optimizer.zero_grad()
           initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
           logits, _ = self.model(inputs, states=initial_states)  # [B, num_classes, iterations]
          
           # Compute loss on the last timestep prediction
           last_logits = logits[:, :, -1]  # [B, num_classes]
           last_targets = targets[:, -1]  # [B]
           loss = self.loss_fn(last_logits, last_targets)

           predictions = last_logits.argmax(dim=1)
           accuracy = (predictions == last_targets).float().mean().item()
  
           loss.backward()
  
           grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
           if self.max_grad_norm > 0:
               torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

           self.optimizer.step()
           objectives.append((loss.item(), grad_norm.item(), accuracy))
           pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{accuracy:.4f}"})
      
       self.scheduler.step()
       objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]  # Calculate average objective values over training
       metrics = {"loss": objectives[0], "grad_norm": objectives[1], "accuracy": objectives[2], "lr": self.scheduler.get_last_lr()[0]}
       return metrics

   def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
       """Internal evaluation function used by both eval() and test()"""
       self.model.eval()
       total_loss = 0
       total_accuracy = 0

       for inputs, targets in tqdm(dataloader, leave=False, desc=desc):
           with torch.inference_mode():
               inputs = inputs.to(self.device)
               targets = targets.to(self.device)
               initial_states = self.model.get_initial_states(batch_size=inputs.size(0))
               logits, _ = self.model(inputs, states=initial_states)

               # Compute loss on the last timestep prediction
               last_logits = logits[:, :, -1]
               last_targets = targets[:, -1]
               loss = self.loss_fn(last_logits, last_targets)

               total_loss += loss.item()

               predictions = last_logits.argmax(dim=1)
               accuracy = (predictions == last_targets).float().mean().item()
               total_accuracy += accuracy

       num_batches = len(dataloader)
       return {
           "loss": total_loss / num_batches,
           "accuracy": total_accuracy / num_batches
       }

   def validate(self, epoch: int) -> dict[str, float]:
       return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

   def test(self, epoch: int = None) -> dict[str, float]:
       desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
       return self._evaluate(self.test_dataloader, desc)

   def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
       accuracies = [pair[1] for pair in metrics["eval_accuracy"]]
       recent_accuracies = accuracies[-window_size:]
       return sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

   def calculate_accuracy(self, predictions, info, targets) -> float:
       """Calculate accuracy from predictions and targets.
      
       For standard image classification, we compute accuracy on the last timestep.
       """
       # predictions shape: [B, num_classes, iterations]
       # targets shape: [B, iterations]
       last_predictions = predictions[:, :, -1].argmax(dim=1)  # [B]
       last_targets = targets[:, -1]  # [B]
       accuracy = (last_predictions == last_targets).float().mean().item()
       return accuracy

   def init_lazy_modules(self):
       B = 1
       img_size = self.dataset_config['image_size']
       num_channels = self.dataset_config['num_channels']       
       input_shape = (B, self.iterations, num_channels, img_size, img_size)
       pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
       initial_states = self.model.get_initial_states(batch_size=B)
       self.model(pseudo_inputs, states=initial_states)
