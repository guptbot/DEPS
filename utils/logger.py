import wandb

# Centralized logging interface for experiment tracking
class Logger:
    @staticmethod
    def init(project, config, dir, name, entity=None, mode=None):
        """Initialize wandb logging"""
        if mode:
            wandb.init(mode=mode)
        else:
            wandb.init(project=project, config=config, dir=dir, name=name, entity=entity)

    @staticmethod
    def log(data, step=None):
        """Log metrics to wandb"""
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)

    @staticmethod
    def define_metric(name, step_metric=None, hidden=False):
        """Define a custom metric for wandb"""
        if step_metric:
            wandb.define_metric(name, step_metric=step_metric)
        else:
            wandb.define_metric(name, hidden=hidden)

    @staticmethod
    def create_image(data, caption=None):
        """Create a wandb Image object"""
        return wandb.Image(data, caption=caption)

    @staticmethod
    def create_video(data, fps=40, format="mp4"):
        """Create a wandb Video object"""
        return wandb.Video(data, fps=fps, format=format)
