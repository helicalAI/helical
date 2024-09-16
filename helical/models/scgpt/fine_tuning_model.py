import torch
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalRNAModel

class scGPTFineTuningModel(torch.nn.Module):
    """Fine-tuning model for the scGPT model.

    Parameters
    ----------
    helical_model : scGPT
        The initialised scGPT model to fine-tune.
    fine_tuning_head : HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model.

    Methods
    -------
    forward(input_gene_ids: torch.Tensor, data_dict: dict, src_key_padding_mask: torch.Tensor, use_batch_labels: bool, device: str) -> torch.Tensor
        The forward method of the fine-tuning model.
    """
    def __init__(self, scGPT_model: HelicalRNAModel, fine_tuning_head: HelicalBaseFineTuningHead):
        super(scGPTFineTuningModel, self).__init__()
        self.helical_model = scGPT_model
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