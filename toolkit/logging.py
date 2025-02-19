from datetime import datetime
from typing import OrderedDict, Optional
from PIL import Image
import numpy as np
from toolkit.config_modules import LoggingConfig

# Base logger class
# This class does nothing, it's just a placeholder
class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    # start logging the training
    def start(self):
        pass
    
    # collect the log to send
    def log(self, *args, **kwargs):
        pass
    
    # send the log
    def commit(self, step: Optional[int] = None):
        pass

    # log image
    def log_image(self, *args, **kwargs):
        pass

    # finish logging
    def finish(self):
        pass

class TensorBoardLogger(EmptyLogger):
    def __init__(self, log_dir: str | None = None, config: OrderedDict = None, *args, **kwargs) -> None:
        """
        Initialize TensorBoard logger
        Args:
            log_dir: Directory where to save the log files. If None, creates a default path
            config: Configuration dictionary to be logged
        """
        super().__init__(*args, **kwargs)

        self.log_dir = log_dir or f"runs/tensorboard_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.config = config
        self._writer = None
        self._current_step = 0

    def start(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("Failed to import tensorboard. Please install it by running `pip install tensorboard`")

        self._writer = SummaryWriter(log_dir=self.log_dir)

        # Log config as text if provided
        if self.config:
            config_str = "\n".join([f"{k}: {v}" for k, v in self.config.items()])
            self._writer.add_text("config", config_str, 0)

    def log(self, metrics: dict, **kwargs):
        """
        Log metrics to TensorBoard
        Args:
            metrics: Dictionary of metric names and values to log
        """
        if not self._writer:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(name, value, self._current_step)
            elif isinstance(value, (list, np.ndarray)):
                self._writer.add_histogram(name, value, self._current_step)

    def commit(self, step: Optional[int] = None):
        """Update the step counter"""
        if step is not None:
            self._current_step = step
        else:
            self._current_step += 1

    def log_image(
            self,
            image: Image,
            id,
            caption: str | None = None,
            *args,
            **kwargs,
    ):
        """
        Log an image to TensorBoard
        Args:
            image: PIL Image to log
            id: Sample index
            caption: Optional caption for the image
        """
        if not self._writer:
            return

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Add caption as text if provided
        if caption:
            self._writer.add_text(f"sample_{id}_caption", caption, self._current_step)

        # Log the image
        self._writer.add_image(f"sample_{id}", img_array, self._current_step, dataformats='HWC')

    def finish(self):
        """Close the TensorBoard writer"""
        if self._writer:
            self._writer.close()
            self._writer = None


# Wandb logger class
# This class logs the data to wandb
class WandbLogger(EmptyLogger):
    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config

    def start(self):
        try:
            import wandb
        except ImportError:
            raise ImportError("Failed to import wandb. Please install wandb by running `pip install wandb`")
        
        # send the whole config to wandb
        run = wandb.init(project=self.project, name=self.run_name, config=self.config)
        self.run = run
        self._log = wandb.log # log function
        self._image = wandb.Image # image object

    def log(self, *args, **kwargs):
        # when commit is False, wandb increments the step,
        # but we don't want that to happen, so we set commit=False
        self._log(*args, **kwargs, commit=False)

    def commit(self, step: Optional[int] = None):
        # after overall one step is done, we commit the log
        # by log empty object with commit=True
        self._log({}, step=step, commit=True)

    def log_image(
        self,
        image: Image,
        id,  # sample index
        caption: str | None = None,  # positive prompt
        *args,
        **kwargs,
    ):
        # create a wandb image object and log it
        image = self._image(image, caption=caption, *args, **kwargs)
        self._log({f"sample_{id}": image}, commit=False)

    def finish(self):
        self.run.finish()

# create logger based on the logging config
def create_logger(logging_config: LoggingConfig, all_config: OrderedDict):
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        return WandbLogger(project=project_name, run_name=run_name, config=all_config)
    elif logging_config.use_tensorboard:  # Add this condition
        print(">>>> Using Tensorboard <<<<")
        return TensorBoardLogger(log_dir=logging_config.tensorboard_log_dir, config=all_config)
    else:
        return EmptyLogger()

