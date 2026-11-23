"""Tasks used in the paper and its appendix."""

from omegaconf import DictConfig

from tasks.task import Task
from tasks.copy import CopyTask
from tasks.few_shot_cifar import FewShotCIFARTask
from tasks.few_shot_miniimagenet import FewShotMiniImageNetTask
from tasks.few_shot_regression import FewShotRegressionTask
from tasks.sequential_split_mnist import SequentialSplitMNISTTask
from tasks.image_classification_cifar10 import ImageClassificationCIFAR10Task
from tasks.rl import MazeExploration


_TASKS = {
    "Copy": CopyTask,
    "FewShotCIFAR": FewShotCIFARTask,
    "FewShotMiniImageNet": FewShotMiniImageNetTask,
    "FewShotRegression": FewShotRegressionTask,
    "SequentialSplitMNIST": SequentialSplitMNISTTask,
    "MazeExploration": MazeExploration,
    "ImageClassificationCIFAR10": ImageClassificationCIFAR10Task,
}


def construct_task(cfg: DictConfig, task_type: str) -> Task:
    return _TASKS[task_type](cfg)
