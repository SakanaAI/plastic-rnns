import torch
import torch.nn as nn
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

from models import construct_rnn
from tasks.task import Task
from tasks.utils import get_device

class CopyModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str):
        super(CopyModel, self).__init__()
        self.rnn = construct_rnn(d_input=task_cfg.copy_dim + 2, d_model=model_cfg.d_model, model_type=model_type, model_kwargs=model_cfg)
        self.output_layer = nn.Linear(model_cfg.d_model, task_cfg.copy_dim)

    def forward(self, input):
        hidden, _ = self.rnn(input, state=self.rnn.get_initial_state())
        logits = self.output_layer(hidden)
        return logits, hidden

class CopyTask(Task):
    @staticmethod
    def build_model(cfg, model_type):
        model = CopyModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type)
        with torch.no_grad():
            model(torch.zeros(1, 1, cfg.task.copy_dim + 2))
        return model

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = get_device(cfg)
        self.copy_dim = cfg.task.copy_dim
        self.copy_sequence_length = cfg.task.copy_sequence_length
        self.copy_delay = cfg.task.copy_delay

        train_data = CopyDataset(dim=cfg.task.copy_dim, seq_length=cfg.task.copy_sequence_length, delay=cfg.task.copy_delay)
        val_data = CopyDataset(dim=cfg.task.copy_dim, seq_length=cfg.task.copy_sequence_length, delay=cfg.task.copy_delay)
        test_data = CopyDataset(dim=cfg.task.copy_dim, seq_length=cfg.task.copy_sequence_length, delay=cfg.task.copy_delay)

        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg.training.batch_size, shuffle=True, drop_last=False)

        model_type = HydraConfig.get().runtime.choices.model
        self.model = CopyModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type).to(self.device)
        self.loss = MSELoss()
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
        self.init_lazy_modules()
        print(f"Total Parameter Count: {self.get_parameter_count()}")

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        objectives = []
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.transpose(1, 0)  # (B, T, D) -> (T, B, D)
            self.optimizer.zero_grad()
            logits, _ = self.model(inputs)
            logits = logits.transpose(1, 0)  # (T, B, D) -> (B, T, D)
            loss = self.loss(logits[:, self.copy_sequence_length + self.copy_delay:, :], targets[:, self.copy_sequence_length + self.copy_delay:, :])
            loss.backward()
    
            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))            
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            objectives.append((loss.item(), grad_norm.item()))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        self.scheduler.step()
        objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]  # Calculate average objective values over training
        metrics = {"loss": objectives[0], "grad_norm": objectives[1], "lr": self.scheduler.get_last_lr()[0]}
        return metrics

    def calculate_accuracy(self):
        """Copy task does not use accuracy."""
        pass

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        """Internal evaluation function used by both eval() and test()"""
        self.model.eval()
        total_loss = 0
        for inputs, targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.transpose(1, 0)  # (B, T, D) -> (T, B, D)
                predictions, _ = self.model(inputs)
                predictions = predictions.transpose(1, 0)  # (T, B, D) -> (B, T, D)
                # Only compute loss on the final sequence portion (as per paper)
                loss = self.loss(predictions[:, self.copy_sequence_length + self.copy_delay:, :], targets[:, self.copy_sequence_length + self.copy_delay:, :])
                total_loss += loss.item()
        return {"loss": total_loss / len(dataloader)}

    def validate(self, epoch: int) -> dict[str, float]:
        return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
        return self._evaluate(self.test_dataloader, desc)

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 1) -> float:
        accuracies = [pair[1] for pair in metrics["eval_loss"]]
        recent_accuracies = accuracies[-window_size:]
        return sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

    def init_lazy_modules(self):
        B = 1
        input_shape = ((2 * self.copy_sequence_length) + self.copy_delay, B, self.copy_dim + 2) # +2 for flags
        pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
        self.model(pseudo_inputs)


class CopyDataset(Dataset):

    def __init__(self, dim: int, seq_length: int, delay: int):

        self.dim = dim
        self.seq_length = seq_length
        self.delay = delay

    def __len__(self):
        return 6400

    def __getitem__(self, index):

        seq = torch.empty((self.seq_length, self.dim)).uniform_(-1, 1)
        sample_input = torch.cat((
            seq,
            torch.ones((self.seq_length, 1)),
            torch.zeros((self.seq_length, 1)),
        ), dim=1)

        delay_input = torch.zeros((self.delay, self.dim + 2))
        test_input = torch.zeros((self.seq_length, self.dim + 2))
        test_input[:, -1] = 1

        input = torch.cat((sample_input, delay_input, test_input), dim=0)
        output = torch.cat((torch.zeros((self.seq_length + self.delay, self.dim)), seq), dim=0)

        return input, output

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)
