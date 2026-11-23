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


class FewShotRegressionModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str):
        super().__init__()
        d_input = task_cfg.n_var + 2  # variables + noisy target + mask bit
        self.input_projection = nn.Sequential(nn.Linear(d_input, model_cfg.d_model), nn.ReLU())
        self.rnn = construct_rnn(d_input=model_cfg.d_model, d_model=model_cfg.d_model, model_type=model_type, model_kwargs=model_cfg)
        self.output_layer = nn.Linear(model_cfg.d_model, 1)

    def forward(self, input):
        input = self.input_projection(input)
        hidden, _ = self.rnn(input, state=self.rnn.get_initial_state())
        predictions = self.output_layer(hidden)
        return predictions, hidden


class FewShotRegressionTask(Task):
    @staticmethod
    def build_model(cfg, model_type):
        model = FewShotRegressionModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type)
        with torch.no_grad():
            model(torch.zeros(1, 1, cfg.task.n_var + 2))
        return model

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = get_device(cfg)
        self.n_var = cfg.task.n_var
        self.train_length = cfg.task.train_length
        self.test_length = cfg.task.test_length

        dataset_kwargs = dict(
            n_var=cfg.task.n_var,
            train_length=cfg.task.train_length,
            test_length=cfg.task.test_length,
            input_noise=cfg.task.input_noise,
            task_mode=cfg.task.task_mode,
            dataset_size=cfg.task.dataset_size,
        )
        train_data = FewShotRegressionDataset(**dataset_kwargs)
        val_data = FewShotRegressionDataset(**dataset_kwargs)
        test_data = FewShotRegressionDataset(**dataset_kwargs)

        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg.training.batch_size, shuffle=True, drop_last=False)

        model_type = HydraConfig.get().runtime.choices.model
        self.model = FewShotRegressionModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type).to(self.device)
        self.loss = nn.MSELoss()
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
            predictions, _ = self.model(inputs)
            predictions = predictions.transpose(1, 0)  # (T, B, 1) -> (B, T, 1)
            # Loss over all timesteps (support + query), matching the paper
            loss = self.loss(predictions, targets)
            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            objectives.append((loss.item(), grad_norm.item()))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        self.scheduler.step()
        objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]
        metrics = {"loss": objectives[0], "grad_norm": objectives[1], "lr": self.scheduler.get_last_lr()[0]}
        return metrics

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        for inputs, targets in tqdm(dataloader, leave=False, desc=desc):
            with torch.inference_mode():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.transpose(1, 0)  # (B, T, D) -> (T, B, D)
                predictions, _ = self.model(inputs)
                predictions = predictions.transpose(1, 0)  # (T, B, 1) -> (B, T, 1)
                loss = self.loss(predictions, targets)
                total_loss += loss.item()
        return {"loss": total_loss / len(dataloader)}

    def validate(self, epoch: int) -> dict[str, float]:
        return self._evaluate(self.val_dataloader, f"Val Epoch {epoch}")

    def test(self, epoch: int = None) -> dict[str, float]:
        desc = f"Test Epoch {epoch}" if epoch is not None else "Test"
        return self._evaluate(self.test_dataloader, desc)

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 1) -> float:
        losses = [pair[1] for pair in metrics["eval_loss"]]
        recent_losses = losses[-window_size:]
        return -(sum(recent_losses) / len(recent_losses)) if recent_losses else 0.0

    def init_lazy_modules(self):
        B = 1
        T = self.train_length + self.test_length
        pseudo_inputs = torch.zeros((T, B, self.n_var + 2), device=self.device).float()
        self.model(pseudo_inputs)


class FewShotRegressionDataset(Dataset):

    def __init__(self, n_var: int, train_length: int, test_length: int, input_noise: float, task_mode: str, dataset_size: int = 10000):
        self.n_var = n_var
        self.train_length = train_length
        self.test_length = test_length
        self.input_noise = input_noise
        self.task_mode = task_mode
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.task_mode == 'linear':
            f = nn.Linear(self.n_var, 1)
        elif self.task_mode == 'mlp':
            f = nn.Sequential(
                nn.Linear(self.n_var, 2),
                nn.Tanh(),
                nn.Linear(2, 1),
            )
        else:
            raise ValueError(f"Unknown task_mode: {self.task_mode}")
        f.requires_grad_(False)

        variables = torch.rand((self.train_length + self.test_length, self.n_var)) * 2 - 1

        values = f(variables)
        values = (values - values.mean()) / values.std()

        input = torch.zeros((self.train_length + self.test_length, self.n_var + 2))
        input[:, :self.n_var] = variables
        # Support samples: noisy target + mask=1
        input[:self.train_length, -2:-1] = values[:self.train_length] + torch.randn_like(values[:self.train_length]) * self.input_noise
        input[:self.train_length, -1:] = 1

        return input, values
