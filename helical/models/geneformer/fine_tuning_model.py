from typing import Literal, Optional
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalBaseFineTuningModel
from helical.models.geneformer import Geneformer, GeneformerConfig
import torch
from torch import optim
from torch.nn.modules import loss
from helical.models.geneformer.geneformer_utils import gen_attention_mask, get_model_input_size, pad_tensor_list
from datasets import Dataset
import logging
from tqdm import trange
from transformers import get_scheduler


logger = logging.getLogger(__name__)

class GeneformerFineTuningModel(HelicalBaseFineTuningModel, Geneformer):
    """GeneformerFineTuningModel
    Fine-tuning model for the Geneformer model.
    
    Parameters
    ----------
    geneformer_config : GeneformerConfig
        The Geneformer configs to fine-tune, the same as instantiating the standard Geneformer model.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification", "regression") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
    forward(input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor
        The forward method of the fine-tuning model.
    train(train_dataset: Dataset, optimizer: optim, optimizer_params: dict, loss_function: loss, label: str, epochs: int, freeze_layers: int, validation_dataset: Optional[Dataset], lr_scheduler_params: Optional[dict], silent = False)
        Fine-tunes the Geneformer model.
    get_outputs(dataset: Dataset, silent = False)
        Get outputs from the fine-tuned model on the given processed dataset.
    """
    def __init__(self,
                 geneformer_config: GeneformerConfig, 
                 fine_tuning_head: Literal["classification", "regression"] | HelicalBaseFineTuningHead, 
                 output_size: Optional[int]=None):
        
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        Geneformer.__init__(self, geneformer_config)

        self.fine_tuning_head.set_dim_size(self.config["embsize"])

    def _forward(self, input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the fine-tuning model.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input ids to the fine-tuning model.
        attention_mask_minibatch : torch.Tensor
            The attention mask for the input tensor.
        
        Returns
        -------
        torch.Tensor
            The output tensor of the fine-tuning model.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask_minibatch)
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
            lr_scheduler_params: Optional[dict] = None,
            silent = False):
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
        freeze_layers : int, optional, default = 2
            The number of layers to freeze.
        validation_dataset : Dataset, default = None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        """
                
        model_input_size = get_model_input_size(self.model)

        cls_present = any("<cls>" in key for key in self.gene_token_dict.keys())
        eos_present = any("<eos>" in key for key in self.gene_token_dict.keys())
        if self.emb_mode == "cls":
            if cls_present is False:
                message = "<cls> token missing in token dictionary"
                logger.error(message)
                raise ValueError(message)
            # Check to make sure that the first token of the filtered input data is cls token
            cls_token_id = self.gene_token_dict["<cls>"]
            if cls_token_id != train_dataset["input_ids"][0][0]:
                message = "First token is not <cls> token value"
                logger.error(message)
            assert (
                train_dataset["input_ids"][0][0] == cls_token_id
            ), "First token is not <cls> token value"
        elif self.emb_mode == "cell":
            if cls_present:
                logger.warning(
                    "CLS token present in token dictionary, excluding from average."
                )
            if eos_present:
                logger.warning(
                    "EOS token present in token dictionary, excluding from average."
                )
        
        total_batch_length = len(train_dataset)
        #initialise optimizer
        optimizer = optimizer(self.parameters(), **optimizer_params)

        #initialise lr_scheduler
        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)
        
        if freeze_layers > 0:
            logger.info(f"Freezing the first {freeze_layers} encoder layers of the Geneformer model during fine-tuning.")

            frozen_layers = self.model.bert.encoder.layer[:freeze_layers]

            for module in frozen_layers:
                for param in module.parameters():
                    param.requires_grad = False

        self.to(self.device)

        validation_batch_length = 0
        if validation_dataset is not None:
            validation_batch_length = len(validation_dataset)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            training_loop = trange(0, total_batch_length, self.config["batch_size"], desc="Fine-Tuning", leave=(not silent))
            batch_loss = 0.0
            batches_processed = 0
            for i in training_loop:
                max_range = min(i + self.config["batch_size"], total_batch_length)

                minibatch = train_dataset.select([i for i in range(i, max_range)])
                max_len = int(max(minibatch["length"]))
                minibatch.set_format(type="torch",device=self.device)

                input_data_minibatch = minibatch["input_ids"]
                input_data_minibatch = pad_tensor_list(
                    input_data_minibatch, max_len, self.pad_token_id, model_input_size
                ).to(self.device)

                outputs = self._forward(input_ids=input_data_minibatch, attention_mask_minibatch=gen_attention_mask(minibatch))
                loss = loss_function(outputs, minibatch[label])
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1
                training_loop.set_postfix({"loss": batch_loss/batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")
                optimizer.step()
                optimizer.zero_grad()

                del outputs
                del minibatch
                del input_data_minibatch
            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_dataset is not None:
                testing_loop = trange(0, validation_batch_length, self.config["batch_size"], desc="Fine-Tuning Validation", leave=(not silent))
                val_loss = 0.0
                count = 0.0
                for i in testing_loop:
                    max_range = min(i + self.config["batch_size"], validation_batch_length)

                    minibatch = validation_dataset.select([i for i in range(i, max_range)])
                    max_len = int(max(minibatch["length"]))
                    minibatch.set_format(type="torch",device=self.device)

                    input_data_minibatch = minibatch["input_ids"]
                    input_data_minibatch = pad_tensor_list(
                        input_data_minibatch, max_len, self.pad_token_id, model_input_size
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self._forward(input_ids=input_data_minibatch, attention_mask_minibatch=gen_attention_mask(minibatch))
                    val_loss += loss_function(outputs, minibatch[label]).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss/count})

                    del outputs
                    del minibatch
                    del input_data_minibatch
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(
        self,
        dataset: Dataset,
        silent = False
    ):
        """Predicts the labels for a dataset using the fine-tuned model.

        Parameters
        ----------
        dataset : Dataset
            The processed dataset to generate outputs for.

        Returns
        -------
        np.array
            The predicted labels in the form of a numpy array
        """
        model_input_size = get_model_input_size(self.model)
        self.to(self.device)

        dataset_length = len(dataset)

        cls_present = any("<cls>" in key for key in self.gene_token_dict.keys())
        eos_present = any("<eos>" in key for key in self.gene_token_dict.keys())
        if self.emb_mode == "cls":
            if cls_present is False:
                message = "<cls> token missing in token dictionary"
                logger.error(message)
                raise ValueError(message)
            assert cls_present, "<cls> token missing in token dictionary"
            # Check to make sure that the first token of the filtered input data is cls token
            cls_token_id = self.gene_token_dict["<cls>"]
            if cls_token_id != dataset["input_ids"][0][0]:
                message = "First token is not <cls> token value"
                logger.error(message)
            assert (
                dataset["input_ids"][0][0] == cls_token_id
            ), "First token is not <cls> token value"
        elif self.emb_mode == "cell":
            if cls_present:
                logger.warning(
                    "CLS token present in token dictionary, excluding from average."
                )
            if eos_present:
                logger.warning(
                    "EOS token present in token dictionary, excluding from average."
                )
        
        output = []
        testing_loop = trange(0, dataset_length, self.config["batch_size"], desc="Generating Outputs", leave=(not silent))
        for i in testing_loop:
            max_range = min(i + self.config["batch_size"], dataset_length)

            minibatch = dataset.select([i for i in range(i, max_range)])
            max_len = int(max(minibatch["length"]))
            minibatch.set_format(type="torch",device=self.device)

            input_data_minibatch = minibatch["input_ids"]
            input_data_minibatch = pad_tensor_list(
                input_data_minibatch, max_len, self.pad_token_id, model_input_size
            ).to(self.device)

            with torch.no_grad():
                outputs = self._forward(input_ids=input_data_minibatch, attention_mask_minibatch=gen_attention_mask(minibatch))
                output.append(outputs.clone().detach())
            del outputs
            del minibatch
            del input_data_minibatch

        return torch.cat(output, dim=0).cpu().numpy()