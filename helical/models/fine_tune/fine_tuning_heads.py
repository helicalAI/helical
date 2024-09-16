from helical.models.base_models import HelicalBaseFoundationModel, HelicalBaseFineTuningHead
import torch

class ClassificationHead(HelicalBaseFineTuningHead):
    """Classification Head for fine-tuning a Helical foundation model.

    Parameters
    ----------
    model : HelicalBaseFoundationModel
        The initialised model to fine-tune.
    num_classes : int
        The number of classes to predict.

    Methods
    -------
    forward(inputs: torch.Tensor) -> torch.Tensor
        The forward method of the classification head.

    """
    def __init__(self, model: HelicalBaseFoundationModel, num_classes: int):
        super(HelicalBaseFineTuningHead, self).__init__()
        self.input_size = model.configurer.config["embsize"] if hasattr(model, "configurer") else model.config.embsize
        self.output_size = num_classes
        self.dropout = torch.nn.Dropout(p=0.02)
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        drop = self.dropout(inputs)
        output = self.linear(drop)
        return output