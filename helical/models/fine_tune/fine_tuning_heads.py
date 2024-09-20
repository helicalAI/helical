from helical.models.base_models import HelicalBaseFoundationModel, HelicalBaseFineTuningHead
import torch

class ClassificationHead(HelicalBaseFineTuningHead):
    """Classification Head for fine-tuning a Helical foundation model.

    Parameters
    ----------
    num_classes : int
        The number of classes to predict.
    dropout : float, optional, default=0.02
        The dropout rate to apply to the input tensor before the linear layer.
        
    Methods
    -------
    forward(inputs: torch.Tensor) -> torch.Tensor
        The forward method of the classification head.

    """
    def __init__(self, num_classes: int, dropout: float = 0.02):
        super(ClassificationHead, self).__init__()
        self.output_size = num_classes
        self.dropout = torch.nn.Dropout(p=dropout)
        
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward method of the classification head.
        
        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor to the classification head.

        Returns
        -------
        torch.Tensor
            The output tensor of the classification head.
        """
        drop = self.dropout(inputs)
        output = self.linear(drop)
        return output

    def set_dim_size(self, dim_size: int) -> None:
        """Set the dimension size of the input tensor.
        
        Parameters
        ----------
        dim_size : int
            The dimension size of the input tensor.
        """
        self.linear = torch.nn.Linear(dim_size, self.output_size)