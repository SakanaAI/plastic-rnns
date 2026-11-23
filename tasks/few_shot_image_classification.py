from contextlib import nullcontext

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

from models import construct_rnn
from tasks.task import Task
from tasks.utils import get_device
from models.backbones import FewShotImageClassificationBackbone
from models.meta_learning import MAML, ProtoNet


META_LEARNING_MODELS = {"MAML", "ProtoNet"}


class FewShotImageClassificationModel(nn.Module):
    def __init__(self, model_cfg: DictConfig, task_cfg: DictConfig, model_type: str):
        super().__init__()
        self.backbone = FewShotImageClassificationBackbone(d_input=model_cfg.d_input)
        self.rnn = construct_rnn(d_input=model_cfg.d_input, d_model=model_cfg.d_model, model_type=model_type, model_kwargs=model_cfg)
        self.output_proj = nn.Linear(model_cfg.d_model, task_cfg.out_dims)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: add back proper parameter resetting
        pass

    def forward(self, input, aux_input):
        assert input.ndim == 5, "Input should be of size T x B x C x H x W"
        assert aux_input.ndim == 3, "Aux input should be of size T x B x nKnovel"

        T, B, C, H, W = input.shape
        input = input.view(T * B, C, H, W)
        aux_input = aux_input.view(T * B, -1)

        embedding = self.backbone(input, aux_input)
        embedding = embedding.view(T, B, -1)
        hidden, _ = self.rnn(embedding, state=self.rnn.get_initial_state())
        logits = self.output_proj(hidden)
        return logits

class FewShotImageClassificationTask(Task):
    @staticmethod
    def build_model(cfg, model_type):
        dataset = cfg.task.get('dataset', 'CIFAR100')
        img_size = DATASET_CONFIGS[dataset]['image_size']
        if model_type == "MAML":
            return MAML(
                num_classes=cfg.task.out_dims,
                image_size=img_size,
                hidden_channels=cfg.model.hidden_channels,
                inner_learning_rate=cfg.model.inner_learning_rate,
                inner_steps=cfg.model.inner_steps,
                first_order=cfg.model.first_order,
            )
        if model_type == "ProtoNet":
            return ProtoNet(
                num_classes=cfg.task.out_dims,
                hidden_channels=cfg.model.hidden_channels,
                temperature=cfg.model.temperature,
            )

        model = FewShotImageClassificationModel(model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type)
        nKnovel = cfg.task.get('nKnovel', 5)
        with torch.no_grad():
            model(torch.zeros(1, 1, 3, img_size, img_size), aux_input=torch.zeros(1, 1, nKnovel))
        return model

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = get_device(cfg)

        train_base_dataset = FewShotDataset(dataset_name=cfg.task.dataset, phase='train')
        val_base_dataset = FewShotDataset(dataset_name=cfg.task.dataset, phase='val')
        test_base_dataset = FewShotDataset(dataset_name=cfg.task.dataset, phase='test')

        self.train_dataloader = FewShotDataloader(
            dataset=train_base_dataset,
            internal_ticks=cfg.task.iterations,
            nKnovel=cfg.task.nKnovel,
            nKbase=cfg.task.nKbase,
            nExemplars=cfg.task.nExemplars,
            nTestNovel=cfg.task.nTestNovel,
            nTestBase=cfg.task.nTestBase,
            epoch_size=cfg.training.batch_size * 1000,
            batch_size=cfg.training.batch_size
        )
        self.val_dataloader = FewShotDataloader(
            dataset=val_base_dataset,
            internal_ticks=cfg.task.iterations,
            nKnovel=cfg.task.nKnovel,
            nKbase=cfg.task.nKbase,
            nExemplars=cfg.task.nExemplars,
            nTestNovel=cfg.task.nTestNovel,
            nTestBase=cfg.task.nTestBase,
            epoch_size=cfg.training.batch_size * 25,
            batch_size=cfg.training.batch_size
        )
        self.test_dataloader = FewShotDataloader(
            dataset=test_base_dataset,
            internal_ticks=cfg.task.iterations,
            nKnovel=cfg.task.nKnovel,
            nKbase=cfg.task.nKbase,
            nExemplars=cfg.task.nExemplars,
            nTestNovel=cfg.task.nTestNovel,
            nTestBase=cfg.task.nTestBase,
            epoch_size=cfg.training.batch_size * 25,
            batch_size=cfg.training.batch_size
        )
        model_type = HydraConfig.get().runtime.choices.model
        self.model = self.build_model(cfg, model_type).to(self.device)
        if model_type not in META_LEARNING_MODELS:
            self.init_lazy_modules()
        self.loss = FewShotLoss(
            iterations_per_image=cfg.task.iterations, 
            num_test_images=cfg.task.nTestNovel+cfg.task.nTestBase, 
            num_classes=cfg.task.out_dims
        )
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
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

    def init_lazy_modules(self):
        B, T = 1, self.train_dataloader.internal_ticks
        total_internal_ticks = T * (self.train_dataloader.nKnovel+self.train_dataloader.nKbase+self.train_dataloader.nTestNovel+self.train_dataloader.nTestBase)
        
        img_size = self.train_dataloader.dataset.config['image_size']
        input_shape = (total_internal_ticks, B, 3, img_size, img_size)
        aux_input_shape = (total_internal_ticks, B, self.train_dataloader.nKnovel)
        
        aux_input = torch.zeros(aux_input_shape, device=self.device).float()
        pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
        self.model(pseudo_inputs, aux_input=aux_input)

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        objectives = []
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for (inputs, aux_inputs), targets in pbar:

            inputs = inputs.transpose(0, 1).contiguous() # (B, T, C, H, W) -> (T, B, C, H, W)
            aux_inputs = aux_inputs.transpose(0, 1).contiguous() # (B, T, nKnovel) -> (T, B, nKnovel)
            targets = targets.transpose(0, 1).contiguous() # (B, T) -> (T, B)

            inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            logits= self.model(inputs, aux_input=aux_inputs)
            loss, info = self.loss(logits, targets)

            train_accuracy = self.calculate_accuracy(logits, info, targets)
            train_accuracy_all_images = self.calculate_accuracy_all_images(logits, info, targets)

            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            objectives.append((loss.item(), grad_norm.item(), train_accuracy, train_accuracy_all_images))
            pbar.set_postfix({"loss": f"{loss.item():.4f}","accuracy": f"{train_accuracy:.4f}", "accuracy_all_images": f"{train_accuracy_all_images:.4f}"})
        self.scheduler.step()
        objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]  # Calculate average objective values over training
        metrics = {"loss": objectives[0], "grad_norm": objectives[1], "accuracy": objectives[2], "accuracy_all_images": objectives[3], "lr": self.scheduler.get_last_lr()[0]}
        return metrics

    def calculate_accuracy(self, predictions, info, targets):
        answer_timestep = info["prediction_timesteps"][0]
        targets = targets[answer_timestep, :]
        predictions = predictions[answer_timestep, :, :].argmax(1)
        accuracy = (predictions == targets).float().mean().item()
        return accuracy

    def calculate_accuracy_all_images(self, predictions, info, targets):
        answer_timesteps = info["prediction_timesteps"]
        timesteps_tensor = torch.tensor(answer_timesteps, device=predictions.device)
        targets_at_timesteps = targets[timesteps_tensor, :]
        predictions_at_timesteps = predictions[timesteps_tensor, :, :].argmax(dim=2)
        accuracy = (predictions_at_timesteps == targets_at_timesteps).float().mean().item()
        return accuracy

    def _evaluate(self, dataloader, desc: str) -> dict[str, float]:
        """Internal evaluation function used by both eval() and test()"""
        self.model.eval()
        total_loss, accuracy, accuracy_all_images = 0, 0, 0
        for (inputs, aux_inputs), targets in tqdm(dataloader, leave=False, desc=desc):
            evaluation_context = (
                nullcontext()
                if getattr(self.model, "requires_inner_grad", False)
                else torch.inference_mode()
            )
            with evaluation_context:
                inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
                inputs = inputs.transpose(0, 1).contiguous() # (B, T, C, H, W) -> (T, B, C, H, W)
                aux_inputs = aux_inputs.transpose(0, 1).contiguous() # (B, T, nKnovel) -> (T, B, nKnovel)
                targets = targets.transpose(0, 1).contiguous() # (B, T) -> (T, B)
                predictions = self.model(inputs, aux_input=aux_inputs)
                loss, info = self.loss(predictions, targets)
                total_loss += loss.item()
                accuracy += self.calculate_accuracy(predictions, info, targets)
                accuracy_all_images += self.calculate_accuracy_all_images(predictions, info, targets)
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy / len(dataloader),
            "accuracy_all_images": accuracy_all_images / len(dataloader)
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

class FewShotLoss(nn.Module):
    def __init__(self, iterations_per_image, num_test_images, num_classes):
        super().__init__()
        self.iterations_per_image = iterations_per_image
        self.num_test_images = num_test_images
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        
        predictions = predictions[-(self.iterations_per_image * self.num_test_images):, :, :]
        predictions = predictions.reshape(-1, self.num_classes)

        targets = targets[-(self.iterations_per_image * self.num_test_images):,:]
        targets = targets.reshape(-1)

        loss = nn.CrossEntropyLoss()(predictions, targets)

        info = {
            'prediction_timesteps': [((self.iterations_per_image * self.num_test_images) + (self.iterations_per_image * i + (self.iterations_per_image - 1))) for i in range(self.num_test_images)]
        }

        return loss, info


import os
import numpy as np
import random
import pickle
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Picklable transform classes for multiprocessing
class _ToNumpyArray:
    """Convert PIL Image to numpy array. Picklable replacement for lambda."""
    def __call__(self, x):
        return np.asarray(x)


class _ToTensorCopy(torchvision.transforms.ToTensor):
    """ToTensor with explicit copy for thread safety."""
    def __call__(self, pic):
        return super().__call__(np.array(pic, copy=True))


DATASET_CONFIGS = {
    'CIFAR100': {
        'dir': os.path.join(_PROJECT_ROOT, 'data', 'cifar_fs'),
        'mean': [129.37731888, 124.10583864, 112.47758569],
        'std': [68.20947949, 65.43124043, 70.45866994],
        'image_size': 32,
        'padding': 4,
        'train_file': 'CIFAR_FS_train.pickle',
        'val_file': 'CIFAR_FS_val.pickle',
        'test_file': 'CIFAR_FS_test.pickle',
    },
    'MiniImageNet': {
        'dir': os.path.join(_PROJECT_ROOT, 'data', 'miniImageNet'),
        'mean': [120.39586422, 115.59361427, 104.54012653],
        'std': [70.68188272, 68.27635443, 72.54505529],
        'image_size': 84,
        'padding': 8,
        'train_file': 'miniImageNet_category_split_train_phase_train.pickle',
        'val_file': 'miniImageNet_category_split_val.pickle',
        'test_file': 'miniImageNet_category_split_test.pickle',
        'train_val_file': 'miniImageNet_category_split_train_phase_val.pickle',
        'train_test_file': 'miniImageNet_category_split_train_phase_test.pickle',
    }
}

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

class FewShotDataset(data.Dataset):
    def __init__(self, dataset_name, phase='train', do_not_use_random_transf=False):
        assert dataset_name in DATASET_CONFIGS, f"Dataset {dataset_name} not supported"
        assert phase in ['train', 'val', 'test'], f"Phase {phase} not valid"
        
        self.dataset_name = dataset_name
        self.phase = phase
        self.name = f'{dataset_name}_{phase}'
        self.config = DATASET_CONFIGS[dataset_name]
        
        if phase == 'train':
            data_train = load_data(os.path.join(self.config['dir'], self.config['train_file']))
            self.data = data_train['data']
            self.labels = data_train['labels']
            
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
            
        elif phase in ['val', 'test']:
            if dataset_name == 'CIFAR100':
                data_base = load_data(os.path.join(self.config['dir'], self.config['train_file']))
                novel_file = self.config['test_file'] if phase == 'test' else self.config['val_file']
                data_novel = load_data(os.path.join(self.config['dir'], novel_file))
            else:  # miniImageNet
                base_file = self.config['train_test_file'] if phase == 'test' else self.config['train_val_file']
                data_base = load_data(os.path.join(self.config['dir'], base_file))
                novel_file = self.config['test_file'] if phase == 'test' else self.config['val_file']
                data_novel = load_data(os.path.join(self.config['dir'], novel_file))
            
            self.data = np.concatenate([data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']
            
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            
            self.labelIds_base = list(buildLabelIndex(data_base['labels']).keys())
            self.labelIds_novel = list(buildLabelIndex(data_novel['labels']).keys())
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0, "Base and novel categories should not overlap"
        
        mean_pix = [x/255.0 for x in self.config['mean']]
        std_pix = [x/255.0 for x in self.config['std']]
        self.normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (phase in ['test', 'val']) or do_not_use_random_transf:
            self.transform = transforms.Compose([
                _ToNumpyArray(),
                _ToTensorCopy(),
                self.normalize
            ])
        else:
            img_size = self.config['image_size']
            padding = self.config['padding']
            self.transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                _ToNumpyArray(),
                _ToTensorCopy(),
                self.normalize
            ])
    
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 internal_ticks=10,
                 nKnovel=5,
                 nKbase=-1,
                 nExemplars=1,
                 nTestNovel=15*5,
                 nTestBase=15*5,
                 batch_size=1,
                 num_workers=8,
                 epoch_size=2000):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.internal_ticks = internal_ticks

        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            Kbase = sorted(self.sampleCategories('base', nKbase))
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        Tbase = []
        if len(Kbase) > 0:
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)
        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples, num_classes=None, is_support=False):
        if len(examples) == 0:
            img_size = self.dataset.config['image_size']
            return torch.empty(0, 3, img_size, img_size), torch.empty(0, 0), torch.empty(0, dtype=torch.long)

        images_list, onehots_list, labels_list = [], [], []

        if num_classes is None:
            num_classes = max(label for _, label in examples) + 1

        img_size = self.dataset.config['image_size']

        for img_idx, label in examples:
            img, _ = self.dataset[img_idx]  # [C, H, W]
            repeated_img = img.unsqueeze(0).repeat(self.internal_ticks, 1, 1, 1)

            if is_support:
                onehot = torch.zeros(self.internal_ticks, num_classes, dtype=torch.float32)
                onehot[:, label] = 1.0
            else:
                onehot = torch.zeros(self.internal_ticks, num_classes, dtype=torch.float32)

            repeated_label = torch.full((self.internal_ticks,), label, dtype=torch.long)

            images_list.append(repeated_img)
            onehots_list.append(onehot)
            labels_list.append(repeated_label)

        images = torch.cat(images_list, dim=0)
        onehots = torch.cat(onehots_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return images, onehots, labels

    def load_function(self, iter_idx):
        """Load function for data loader - must be a method for pickling with multiprocessing."""
        Exemplars, Test, Kall, nKbase = self.sample_episode()
        num_classes = len(Kall)

        Xt, Yt_onehot, Yt_labels = self.createExamplesTensorData(Test, num_classes=num_classes, is_support=False)

        if len(Exemplars) > 0:
            Xe, Ye_onehot, Ye_labels = self.createExamplesTensorData(Exemplars, num_classes=num_classes, is_support=True)
            inputs = torch.cat([Xe, Xt], dim=0)
            aux_inputs = torch.cat([Ye_onehot, Yt_onehot], dim=0)
            targets = torch.cat([Ye_labels, Yt_labels], dim=0)
        else:
            inputs, aux_inputs, targets = Xt, Yt_onehot, Yt_labels

        return (inputs, aux_inputs), targets

    def get_iterator(self, epoch=0):
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=self.load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __iter__(self):
        return iter(self.get_iterator())

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
