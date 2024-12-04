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
    """GeneformerFineTuningModel.
    
    Fine-tuning model for the Geneformer model.

    Example
    ----------
    ```python
    from helical import GeneformerConfig, GeneformerFineTuningModel

    # Prepare the data
    ann_data = ad.read_h5ad("dataset.h5ad")

    # Get the desired label class
    cell_types = list(ann_data.obs.cell_type)

    # Create a dictionary mapping the classes to unique integers for training
    label_set = set(cell_types)
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    # Add this column to the Dataset
    dataset = dataset.add_column('cell_types', cell_types)

    # Create the fine-tuning model
    model_config = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)
    geneformer_fine_tune = GeneformerFineTuningModel(
        geneformer_config=model_config, 
        fine_tuning_head="classification", 
        label="cell_types", 
        output_size=len(label_set)
    )

    # Process the data for training
    dataset = geneformer_fine_tune.process_data(ann_data)

    # Fine-tune
    geneformer_fine_tune.train(train_dataset=dataset)

    # Get outputs of the fine-tuned model
    outputs = geneformer_fine_tune.get_outputs(dataset)

    # Get the embeddings of the fine-tuned model
    embeddings = geneformer_fine_tune.get_embeddings(dataset)
    ```
    
    Parameters
    ----------
    geneformer_config : GeneformerConfig
        The Geneformer configs to fine-tune, the same as instantiating the standard Geneformer model.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification", "regression") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the `fine_tuning_head` is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
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
        """
        Fine-tunes the Geneformer model for classification tasks.

        Parameters
        ----------
        train_dataset : Dataset
            A helical-processed dataset for fine-tuning.

        optimizer : torch.optim.Optimizer, optional, default=torch.optim.AdamW
            The optimizer to be used for training.

        optimizer_params : dict
            Parameters to be passed to the specified optimizer. This dictionary should *NOT* include model parameters.
            Example:
                optimizer_params = {'lr': 0.0001}

        loss_function : torch.nn.loss, optional, default=torch.nn.CrossEntropyLoss()
            The loss function to be used for training.

        label : str, optional, default="cell_types"
            The column in the dataset containing the training labels. Labels should be stored as unique integers per class.

        epochs : int, optional, default=10
            The number of epochs to train the model.

        freeze_layers : int, optional, default=2
            The number of layers to freeze during training.

        validation_dataset : Dataset, optional, default=None
            A helical-processed dataset used for per-epoch validation. If not specified, no validation will be performed.

        lr_scheduler_params : dict, optional, default=None
            Parameters for the learning rate scheduler from the Transformers [`get_scheduler`](https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_scheduler) method. The optimizer should not be included 
            in this dictionary, as it will be inferred from the [`optimizer`](https://pytorch.org/docs/main/optim.html) parameter. If not specified, no learning rate scheduler 
            will be used.
            Example:
                lr_scheduler_params = {'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5}
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
        np.ndarray
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