from typing import Literal, Optional
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalBaseFineTuningModel, HelicalRNAModel
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
import torch
from torch import optim
from torch.nn.modules import loss
from helical.models.geneformer.geneformer_utils import fine_tuning
from datasets import Dataset

class GeneformerFineTuningModel(HelicalBaseFineTuningModel):
    """GeneformerFineTuningModel
    Fine-tuning model for the Geneformer model.
    
    Parameters
    ----------
    helical_model : Geneformer
        The initialised Geneformer model to fine-tune.
    fine_tuning_head : HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
    forward(input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor
        The forward method of the fine-tuning model.
    """
    def __init__(self, geneformer_model: HelicalRNAModel, fine_tuning_head: Literal["classification"]|HelicalBaseFineTuningHead, output_size: Optional[int]=None):
        super(GeneformerFineTuningModel, self).__init__()
        self.config = geneformer_model.config
        self.emb_mode = geneformer_model.emb_mode
        self.pad_token_id = geneformer_model.pad_token_id
        self.device = geneformer_model.device
        self.gene_token_dict = geneformer_model.gene_token_dict
        self.helical_model = geneformer_model.model
        if isinstance(fine_tuning_head, str):
            if fine_tuning_head == "classification":
                if output_size is None:
                    raise ValueError("The output_size must be specified for a classification head.")
                fine_tuning_head = ClassificationHead(output_size)
            else:
                raise ValueError(f"The fine_tuning_head must be a valid HelicalBaseFineTuningHead")
        else:
            fine_tuning_head = fine_tuning_head
        fine_tuning_head.set_dim_size(self.config["embsize"])
        self.fine_tuning_head = fine_tuning_head

    def forward(self, input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor:
        outputs = self.helical_model.forward(input_ids=input_ids, attention_mask=attention_mask_minibatch)
        final_layer = outputs.hidden_states[-1]
        cls_seq = final_layer[:, 0, :]
        final = self.fine_tuning_head(cls_seq)
        return final
    
    def train(
            self,
            train_dataset: Dataset, 
            optimizer: optim = optim.AdamW,
            optimizer_params: dict = {'lr': 0.0001}, 
            loss_function: loss = loss.CrossEntropyLoss(), 
            label: str = "cell_types", 
            epochs: int = 1,
            freeze_layers: int = 2,
            validation_dataset: Optional[Dataset] = None,
            lr_scheduler_params: Optional[dict] = None):
        """Fine-tunes the Geneformer model for classification tasks. 

        Parameters
        ----------

        train_dataset : Dataset
            A helical processed dataset for fine-tuning
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        label : str, optional, default = "cell_types"
            The column in the dataset containing the training labels. These should be stored as unique per class integers.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        freeze_layers : int, optional, default = 0
            The number of layers to freeze.
        validation_dataset : Dataset, default = None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        Returns
        -------
        torch.nn.Module
            The fine-tuned model.
        """

        fine_tuning(
            self,
            train_dataset,
            validation_dataset,
            optimizer,
            optimizer_params,
            loss_function,
            label,
            epochs,
            self.pad_token_id,
            self.config["batch_size"],
            self.device,
            lr_scheduler_params,
            freeze_layers,
            self.emb_mode,
            self.gene_token_dict,
        )