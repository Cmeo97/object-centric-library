import logging
import platform
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Union
import hydra
from omegaconf import OmegaConf
from ignite.engine import Engine, Events
from data.dataset_variants import apply_variant
from data.datasets import make_dataloaders
from abc import abstractmethod
from models.nn_utils import summary_num_params
from utils.logging import logging_wrapper, setup_logging
from utils.paths import CONFIG
from utils.utils import (
    SkipTrainingException,
    add_uuid,
    available_cuda_device_names,
    get_cli_overrides,
    omegaconf_equal,
    set_all_seeds,
)

import torch 
from torch import nn
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.metrics_evaluator import MetricsEvaluator
from models.base_model import BaseModel
from models.utils import TrainCheckpointHandler, infer_model_type
from utils.logging import TBLogger, log_engine_stats
from utils.utils import ExitResubmitException, SkipTrainingException
from dataclasses import dataclass, field


MANDATORY_FIELDS = [
    "loss",  # training loss
    "mask",  # masks for all slots (incl. background if any)
    "slot",  # raw slot reconstructions for all slots (incl. background if any)
    "representation",  # slot representations (only foreground, if applicable)
]


@dataclass
class ForwardPass:
    model: BaseModel
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None

    def __call__(self, batch: dict) -> Tuple[dict, dict]:
        for key in batch.keys():
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        
        output = batch["image"]
        return batch, output



@dataclass(eq=False, repr=False)
class BaseModel(nn.Module):
    name: str
    width: int
    height: int

    # This applies only to object-centric models, but must always be defined.
    num_slots: int

    def __post_init__(self):
        # Run the nn.Module initialization logic before we do anything else. Models
        # should call this post-init at the beginning of their post-init.
        super().__init__()

    @property
    def num_representation_slots(self) -> int:
        """Number of slots used for representation.

        By default, it is equal to the number of slots, but when possible we can
        consider only foreground slots (e.g. in SPACE).
        """
        return self.num_slots

    @property
    @abstractmethod
    def slot_size(self) -> int:
        """Representation size per slot.

        This does not apply to models that are not object-centric, but they should still
        define it in the most sensible possible way.
        """
        ...



@dataclass
class BaseTrainer:
    device: str
    steps: int
    optimizer_config: DictConfig
    clip_grad_norm: Optional[float]
    checkpoint_steps: int
    logloss_steps: int
    logweights_steps: int
    logimages_steps: int
    logvalid_steps: int
    debug: bool
    resubmit_steps: Optional[int]
    resubmit_hours: Optional[float]
    working_dir: Path

    model: BaseModel = field(init=False)
    dataloaders: List[DataLoader] = field(init=False)
    eval_step: ForwardPass = field(init=False)
    trainer: Engine = field(init=False)
    training_start: float = field(init=False)

    def __post_init__(self):
        print('post_init')

    def train_step(self, engine: Engine, batch: dict) -> Tuple[dict]:
  
        batch, output = self.eval_step(batch)
        #self._check_shapes(batch, output)  # check shapes of mandatory items
        print('model_input shape : ',output.shape)
        return batch, output
        #[print('key : ', x,' shape : ', batch[x].shape) for x in batch.keys()]
        #return batch

    def setup(self, model: BaseModel, dataloaders: List[DataLoader]):
        """Adds model and dataloaders to the trainer.

        Overriding methods should call this base method first.

        This method adds model and dataloaders to the Trainer object. It creates
        an evaluation step, the optimizer, and sets up tensorboard, but does not
        create a trainer engine. Anything that goes in the checkpoints must be
        created here. Anything that requires a trainer (e.g. callbacks) must be
        defined in `_setup_training()`.
        """
        assert model.training is True  # don't silently set it to train
        self.model = model
        self.dataloaders = dataloaders       
        self.trainer = Engine(self.train_step)
        self.eval_step = ForwardPass(self.model, self.device)
        ProgressBar().attach(self.trainer)

        # Here we only do training and validation.
        if len(self.dataloaders) < 2:
            raise ValueError("At least 2 dataloaders required (train and validation)")
        self.training_dataloader = self.dataloaders[0]
        self.validation_dataloader = self.dataloaders[1]

        print('setup done')


    def train(self):
        self.trainer.run(
            self.training_dataloader, max_epochs=5) #, epoch_length=self.steps
        #)

    def _check_shapes(self, batch: dict, output: dict):
        bs = batch["image"].shape[0]
        if infer_model_type(self.model.name) == "distributed":
            n_slots = 1
            repr_shape = (bs, self.model.num_slots * self.model.slot_size)
        else:
            n_slots = self.model.num_slots
            repr_shape = (bs, self.model.num_representation_slots, self.model.slot_size)
        c = self.dataloaders[0].dataset.input_channels
        h, w = self.model.height, self.model.width
        # These are the fields in MANDATORY_FIELDS
        assert output["loss"].dim() == 0
        assert output["mask"].shape == (bs, n_slots, 1, h, w)
        assert output["slot"].shape == (bs, n_slots, c, h, w)
        assert output["representation"].shape == repr_shape


@hydra.main(config_path=CONFIG, config_name="train_object_discovery")
@logging_wrapper
def main(config):
    curr_dir = Path.cwd()  # Hydra sets and creates cwd automatically
    setup_logging(log_fname="train.log")
    cli_overrides = get_cli_overrides()
    config = apply_variant(config, cli_overrides=cli_overrides)
    logging.info(f"Running on node '{platform.node()}'")
    logging.info(f"Available cuda devices: {available_cuda_device_names()}")

    assert len(config.data_sizes) == 3, "Need a train/validation/test split."

    logging.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Add UUID after logging the config because we might have to replace it.
    add_uuid(config)

    # Hydra does not delete the folder if existing, so its contents are kept.
    train_config_path = curr_dir / "train_config.yaml"

    load_checkpoint = False

    set_all_seeds(config.seed)

    with open(train_config_path, "w") as f:
        OmegaConf.save(config, f)

    logging.info("Creating model")
    model: BaseModel = hydra.utils.instantiate(config.model).to(config.device)
    logging.info(f"Model summary:\n{model}")
    summary_string, num_params = summary_num_params(model, max_depth=4)
    logging.info(f"Model parameters summary:\n{summary_string}")

    logging.info("Creating data loaders")
    dataloaders = make_dataloaders(
        dataset_config=config.dataset,
        data_sizes=config.data_sizes[:2],  # do not instantiate test set here
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory="cuda" in config.device and config.num_workers > 0,
    )

    logging.info("Creating trainer")
    trainer: BaseTrainer = hydra.utils.instantiate(
        config.trainer,
        device=config.device,
        debug=config.debug,
        working_dir=curr_dir,
    )

    try:
        trainer.setup(model, dataloaders)
    except SkipTrainingException:
        return
    
    logging.info("Training starts")
    trainer.train()
    logging.info("Training completed")




if __name__ == "__main__":
    main()


# Keys dsprites

#key :  color  shape :  torch.Size([64, 6, 3])
#key :  image  shape :  torch.Size([64, 3, 64, 64])
#key :  mask  shape :  torch.Size([64, 6, 1, 64, 64])
#key :  num_actual_objects  shape :  torch.Size([64, 1])
#key :  orientation  shape :  torch.Size([64, 6])
#key :  scale  shape :  torch.Size([64, 6])
#key :  shape  shape :  torch.Size([64, 6, 4])
#key :  visibility  shape :  torch.Size([64, 6, 1])
#key :  x  shape :  torch.Size([64, 6])
#key :  y  shape :  torch.Size([64, 6])
#key :  hue  shape :  torch.Size([64, 6])
#key :  saturation  shape :  torch.Size([64, 6])
#key :  value  shape :  torch.Size([64, 6])
#key :  is_foreground  shape :  torch.Size([64, 6, 1])
#key :  sample_id  shape :  torch.Size([64])
#key :  y_true  shape :  torch.Size([64, 6, 10])
#key :  is_modified  shape :  torch.Size([64, 6])


# Keys Tetrominoes 

#key :  color  shape :  torch.Size([64, 4, 7])
#key :  image  shape :  torch.Size([64, 3, 32, 32])
#key :  mask  shape :  torch.Size([64, 4, 1, 32, 32])
#key :  num_actual_objects  shape :  torch.Size([64, 1])
#key :  shape  shape :  torch.Size([64, 4, 20])
#key :  visibility  shape :  torch.Size([64, 4, 1])
#key :  x  shape :  torch.Size([64, 4])
#key :  y  shape :  torch.Size([64, 4])
#key :  is_foreground  shape :  torch.Size([64, 4, 1])
#key :  sample_id  shape :  torch.Size([64])
#key :  y_true  shape :  torch.Size([64, 4, 29])
#key :  is_modified  shape :  torch.Size([64, 4])

# Keys CLEVER 
#key :  color  shape :  torch.Size([64, 11, 9])
#key :  image  shape :  torch.Size([64, 3, 128, 128])
#key :  mask  shape :  torch.Size([64, 11, 1, 128, 128])
#key :  material  shape :  torch.Size([64, 11, 3])
#key :  num_actual_objects  shape :  torch.Size([64, 1])
#key :  rotation  shape :  torch.Size([64, 11])
#key :  shape  shape :  torch.Size([64, 11, 4])
#key :  size  shape :  torch.Size([64, 11, 3])
#key :  visibility  shape :  torch.Size([64, 11, 1])
#key :  x  shape :  torch.Size([64, 11])
#key :  y  shape :  torch.Size([64, 11])
#key :  z  shape :  torch.Size([64, 11])
#key :  x_2d  shape :  torch.Size([64, 11])
#key :  y_2d  shape :  torch.Size([64, 11])
#key :  z_2d  shape :  torch.Size([64, 11])
#key :  is_foreground  shape :  torch.Size([64, 11, 1])
#key :  sample_id  shape :  torch.Size([64])
#key :  y_true  shape :  torch.Size([64, 11, 21])
#key :  is_modified  shape :  torch.Size([64, 11])


# Keys Shapestacks 

#key :  com  shape :  torch.Size([64, 7, 3])
#key :  euler  shape :  torch.Size([64, 7, 3])
#key :  f  shape :  torch.Size([64])
#key :  h  shape :  torch.Size([64])
#key :  image  shape :  torch.Size([64, 3, 64, 64])
#key :  is_stable  shape :  torch.Size([64])
#key :  l  shape :  torch.Size([64])
#key :  level  shape :  torch.Size([64, 7])
#key :  mask  shape :  torch.Size([64, 7, 1, 64, 64])
#key :  num_actual_objects  shape :  torch.Size([64, 1])
#key :  color  shape :  torch.Size([64, 7, 7])
#key :  shape  shape :  torch.Size([64, 7, 4])
#key :  size  shape :  torch.Size([64, 7, 3])
#key :  subset_name  shape :  torch.Size([64])
#key :  v  shape :  torch.Size([64])
#key :  vcom  shape :  torch.Size([64])
#key :  visibility  shape :  torch.Size([64, 7, 1])
#key :  vpsf  shape :  torch.Size([64])
#key :  w  shape :  torch.Size([64])
#key :  x  shape :  torch.Size([64, 7])
#key :  y  shape :  torch.Size([64, 7])
#key :  z  shape :  torch.Size([64, 7])
#key :  is_foreground  shape :  torch.Size([64, 7, 1])
#key :  sample_id  shape :  torch.Size([64])
#key :  y_true  shape :  torch.Size([64, 7, 14])
#key :  is_modified  shape :  torch.Size([64, 7])


# Keys Objects Room 

#key :  image  shape :  torch.Size([64, 3, 64, 64])
#key :  mask  shape :  torch.Size([64, 7, 1, 64, 64])
#key :  num_actual_objects  shape :  torch.Size([64, 1])
#key :  visibility  shape :  torch.Size([64, 7, 1])
#key :  is_foreground  shape :  torch.Size([64, 7, 1])
#key :  sample_id  shape :  torch.Size([64])
#key :  y_true  shape :  torch.Size([64, 7, 0])
#key :  is_modified  shape :  torch.Size([64, 7])





