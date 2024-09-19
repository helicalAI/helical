from typing import Literal, Optional, get_origin
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
from helical.models.scgpt.data_collator import DataCollator
from helical.models.scgpt.dataset import Dataset
from sklearn.metrics import accuracy_score
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalRNAModel
from helical.models.base_models import HelicalBaseFineTuningModel

class scGPTFineTuningModel(HelicalBaseFineTuningModel):
    """Fine-tuning model for the scGPT model.

    Parameters
    ----------
    helical_model : scGPT
        The initialised scGPT model to fine-tune.
    fine_tuning_head : HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
    forward(input_gene_ids: torch.Tensor, data_dict: dict, src_key_padding_mask: torch.Tensor, use_batch_labels: bool, device: str) -> torch.Tensor
        The forward method of the fine-tuning model.

    """
    def __init__(self, scGPT_model: HelicalRNAModel, fine_tuning_head: Literal["classification"]|HelicalBaseFineTuningHead, output_size: Optional[int]=None):
        super(scGPTFineTuningModel, self).__init__()
        self.config = scGPT_model.config
        self.vocab = scGPT_model.vocab
        self.helical_model = scGPT_model.model
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

    def forward(self, input_gene_ids, data_dict, src_key_padding_mask, use_batch_labels, device) -> torch.Tensor:
        embeddings = self.helical_model._encode(
            input_gene_ids,
            data_dict["expr"].to(device),
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=data_dict["batch_labels"].to(device)
            if use_batch_labels
            else None,
        )
        cls_emb = embeddings[:, 0, :]
        output = self.fine_tuning_head(cls_emb)
        return output
    
    def train(
        self,
        train_input_data: Dataset, 
        train_labels,     
        validation_input_data = None,
        validation_labels = None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {'lr': 0.0001}, 
        loss_function: loss = loss.CrossEntropyLoss(), 
        epochs: int = 1,
        # freeze_layers: int = 0,
        lr_scheduler_params: Optional[dict] = None):
        """Fine-tunes the scGPT model with different head modules. 

        Parameters
        ----------
        train_input_data : Dataset
            A helical scGPT processed dataset for fine-tuning
        train_labels : ndarray
            The labels for the training data. These should be stored as unique per class integers.
        validation_input_data : Dataset, default = None
            A helical scGPT processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : ndarray, default = None
            The labels for the validation data. These should be stored as unique per class integers.
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        freeze_layers : int, optional, default = 0
            The number of layers to freeze.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        Returns
        -------
        torch.nn.Module
            The fine-tuned model.
        """
        
        device = next(self.helical_model.parameters()).device

        try:
            use_batch_labels = train_input_data.batch_ids is not None
        except:
            use_batch_labels = False
                
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )

        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        if validation_input_data is not None:
            validation_data_loader = DataLoader(
                validation_input_data,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(validation_input_data),
                collate_fn=collator,
                drop_last=False,
                pin_memory=True,
            )

        self.to(device)

        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        with torch.cuda.amp.autocast(enabled=True): #torch.autocast(device_type=str(device),enabled=True): # torch.cuda.amp.autocast(enabled=True):
            for j in range(epochs):
                batch_count = 0
                batch_loss = 0.0
                batches_processed = 0
                training_loop = tqdm(data_loader)
                for data_dict in training_loop:
                    input_gene_ids = data_dict["gene"].to(device)
                    src_key_padding_mask = input_gene_ids.eq(
                        self.vocab[self.config["pad_token"]]
                    )
                    output = self(input_gene_ids, data_dict, src_key_padding_mask, use_batch_labels, device)
                    labels = torch.tensor(train_labels[batch_count: batch_count + self.config["batch_size"]], device=device)
                    batch_count += self.config["batch_size"]
                    loss = loss_function(output, labels)
                    loss.backward()
                    batch_loss += loss.item()
                    batches_processed += 1
                    optimizer.step()
                    optimizer.zero_grad()

                    training_loop.set_postfix({"loss": batch_loss/batches_processed})
                    training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

                if lr_scheduler is not None:
                    lr_scheduler.step()

                if validation_input_data is not None:
                    testing_loop = tqdm(validation_data_loader, desc="Fine-Tuning Validation")
                    accuracy = 0.0
                    count = 0.0
                    validation_batch_count = 0
                    for validation_data_dict in testing_loop:
                        input_gene_ids = validation_data_dict["gene"].to(device)
                        src_key_padding_mask = input_gene_ids.eq(
                            self.vocab[self.config["pad_token"]]
                        )
                        output = self(input_gene_ids, validation_data_dict, src_key_padding_mask, use_batch_labels, device)
                        val_labels = torch.tensor(validation_labels[validation_batch_count: validation_batch_count + self.config["batch_size"]], device=device)
                        validation_batch_count += self.config["batch_size"]
                        accuracy += accuracy_score(val_labels.cpu(), torch.argmax(output, dim=1).cpu())
                        count += 1.0
                        testing_loop.set_postfix({"accuracy": accuracy/count})

        # put model back in eval mode
        self.eval()
        # should the model be returned #TODO
        # return self
    