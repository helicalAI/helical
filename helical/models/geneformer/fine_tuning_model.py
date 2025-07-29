from typing import Literal, Optional
from helical.models.base_models import (
    HelicalBaseFineTuningHead,
    HelicalBaseFineTuningModel,
)
from helical.models.geneformer import Geneformer, GeneformerConfig
import torch
from torch import optim
from torch.nn.modules import loss
from helical.models.geneformer.geneformer_utils import (
    gen_attention_mask,
    get_model_input_size,
    pad_tensor_list,
    _check_for_expected_special_tokens,
    mean_nonpadding_embs,
)
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
    from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
    import anndata as ad

    # Load the data
    ann_data = ad.read_h5ad("/home/matthew/helical-dev/helical/yolksac_human.h5ad")

    # Get the column for fine-tuning
    cell_types = list(ann_data.obs["cell_types"])
    label_set = set(cell_types)

    # Create a GeneformerConfig object
    geneformer_config = GeneformerConfig(model_name="gf-12L-38M-i4096", batch_size=10)

    # Create a GeneformerFineTuningModel object
    geneformer_fine_tune = GeneformerFineTuningModel(geneformer_config=geneformer_config, fine_tuning_head="classification", output_size=len(label_set))

    # Process the data
    dataset = geneformer_fine_tune.process_data(ann_data[:10])

    # Add column to the dataset
    dataset = dataset.add_column('cell_types', cell_types)

    # Create a dictionary to map cell types to ids
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    def classes_to_ids(example):
        example["cell_types"] = class_id_dict[example["cell_types"]]
        return example

    # Convert cell types to ids
    dataset = dataset.map(classes_to_ids, num_proc=1)

    # Fine-tune the model
    geneformer_fine_tune.train(train_dataset=dataset, label="cell_types")

    # Get logits from the fine-tuned model
    outputs = geneformer_fine_tune.get_outputs(dataset)
    print(outputs[:10])

    # Get embeddings from the fine-tuned model
    embeddings = geneformer_fine_tune.get_embeddings(dataset)
    print(embeddings[:10])
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

    def __init__(
        self,
        geneformer_config: GeneformerConfig,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: Optional[int] = None,
    ):

        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        Geneformer.__init__(self, geneformer_config)

        self.fine_tuning_head.set_dim_size(self.config["embsize"])

    def _forward(
        self,
        input_ids: torch.tensor,
        attention_mask_minibatch: torch.tensor,
        original_lengths: torch.tensor,
    ) -> torch.tensor:
        """
        Forward method of the fine-tuning model.

        Parameters
        ----------
        input_ids : torch.tensor
            The input ids to the fine-tuning model.
        attention_mask_minibatch : torch.tensor
            The attention mask for the input tensor.
        original_lengths: torch.tensor
            The original lengths of the inputs without padding

        Returns
        -------
        torch.tensor
            The output tensor of the fine-tuning model.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask_minibatch
        )
        batch_embeddings = outputs.hidden_states[-1]

        if self.emb_mode == "cls" and self.cls_present:
            batch_embeddings = batch_embeddings[:, 0, :]
        else:
            length = original_lengths
            if self.cls_present:
                batch_embeddings = batch_embeddings[
                    :, 1:, :
                ]  # Get all layers except the cls embs
                if self.eos_present:
                    length -= 2  # length is used for the mean calculation, 2 is subtracted because we have taken both the cls and eos embeddings out
                else:
                    length -= 1  # length is subtracted because just the cls is removed

            batch_embeddings = mean_nonpadding_embs(batch_embeddings, length)

        final = self.fine_tuning_head(batch_embeddings)
        return final

    def train(
        self,
        train_dataset: Dataset,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        label: str = "cell_types",
        epochs: int = 1,
        freeze_layers: int = 2,
        validation_dataset: Optional[Dataset] = None,
        lr_scheduler_params: Optional[dict] = None,
        silent=False,
    ):
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

        _check_for_expected_special_tokens(
            train_dataset,
            self.emb_mode,
            self.cls_present,
            self.eos_present,
            self.tk.gene_token_dict,
        )

        total_batch_length = len(train_dataset)
        # initialise optimizer
        optimizer = optimizer(self.parameters(), **optimizer_params)

        # initialise lr_scheduler
        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        if freeze_layers > 0:
            logger.info(
                f"Freezing the first {freeze_layers} encoder layers of the Geneformer model during fine-tuning."
            )

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
            training_loop = trange(
                0,
                total_batch_length,
                self.config["batch_size"],
                desc="Fine-Tuning",
                leave=(not silent),
            )
            batch_loss = 0.0
            batches_processed = 0
            for i in training_loop:
                max_range = min(i + self.config["batch_size"], total_batch_length)

                minibatch = train_dataset.select([i for i in range(i, max_range)])
                max_len = int(max(minibatch["length"]))
                minibatch.set_format(type="torch", device=self.device)

                input_data_minibatch = minibatch["input_ids"]
                input_data_minibatch = pad_tensor_list(
                    input_data_minibatch, max_len, self.pad_token_id, model_input_size
                ).to(self.device)

                outputs = self._forward(
                    input_ids=input_data_minibatch,
                    attention_mask_minibatch=gen_attention_mask(minibatch),
                    original_lengths=minibatch["length"],
                )
                loss = loss_function(outputs, minibatch[label])
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1
                training_loop.set_postfix({"loss": batch_loss / batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")
                optimizer.step()
                optimizer.zero_grad()

                del outputs
                del minibatch
                del input_data_minibatch
            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_dataset is not None:
                testing_loop = trange(
                    0,
                    validation_batch_length,
                    self.config["batch_size"],
                    desc="Fine-Tuning Validation",
                    leave=(not silent),
                )
                val_loss = 0.0
                count = 0.0
                for i in testing_loop:
                    max_range = min(
                        i + self.config["batch_size"], validation_batch_length
                    )

                    minibatch = validation_dataset.select(
                        [i for i in range(i, max_range)]
                    )
                    max_len = int(max(minibatch["length"]))
                    minibatch.set_format(type="torch", device=self.device)

                    input_data_minibatch = minibatch["input_ids"]
                    input_data_minibatch = pad_tensor_list(
                        input_data_minibatch,
                        max_len,
                        self.pad_token_id,
                        model_input_size,
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self._forward(
                            input_ids=input_data_minibatch,
                            attention_mask_minibatch=gen_attention_mask(minibatch),
                            original_lengths=minibatch["length"],
                        )
                    val_loss += loss_function(outputs, minibatch[label]).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})

                    del outputs
                    del minibatch
                    del input_data_minibatch
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: Dataset, silent=False):
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
        self.model.eval()
        self.fine_tuning_head.eval()

        model_input_size = get_model_input_size(self.model)
        self.to(self.device)

        dataset_length = len(dataset)

        _check_for_expected_special_tokens(
            dataset,
            self.emb_mode,
            self.cls_present,
            self.eos_present,
            self.tk.gene_token_dict,
        )

        output = []
        testing_loop = trange(
            0,
            dataset_length,
            self.config["batch_size"],
            desc="Generating Outputs",
            leave=(not silent),
        )
        for i in testing_loop:
            max_range = min(i + self.config["batch_size"], dataset_length)

            minibatch = dataset.select([i for i in range(i, max_range)])
            max_len = int(max(minibatch["length"]))
            minibatch.set_format(type="torch", device=self.device)

            input_data_minibatch = minibatch["input_ids"]
            input_data_minibatch = pad_tensor_list(
                input_data_minibatch, max_len, self.pad_token_id, model_input_size
            ).to(self.device)

            with torch.no_grad():
                outputs = self._forward(
                    input_ids=input_data_minibatch,
                    attention_mask_minibatch=gen_attention_mask(minibatch),
                    original_lengths=minibatch["length"],
                )
                output.append(outputs.clone().detach())
            del outputs
            del minibatch
            del input_data_minibatch

        return torch.cat(output, dim=0).cpu().numpy()
