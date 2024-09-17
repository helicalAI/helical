import torch
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalRNAModel

class UCEFineTuningModel(torch.nn.Module):
    """Fine-tuning model for the UCE model.

    Parameters
    ----------
    helical_model : UCE
        The initialised UCE model to fine-tune.
    fine_tuning_head : HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model.

    Methods
    -------
    forward(input_gene_ids: torch.Tensor, data_dict: dict, src_key_padding_mask: torch.Tensor, use_batch_labels: bool, device: str) -> torch.Tensor
        The forward method of the fine-tuning model.
    """
    def __init__(self, uce_model: HelicalRNAModel, fine_tuning_head: HelicalBaseFineTuningHead):
        super(UCEFineTuningModel, self).__init__()
        self.helical_model = uce_model.model
        self.fine_tuning_head = fine_tuning_head

    def forward(self, batch_sentences, mask) -> torch.Tensor:
        _, embeddings = self.helical_model.forward(batch_sentences, mask=mask)
        # if self.helical_model.accelerator is not None:
        #     self.accelerator.wait_for_everyone()
        #     embeddings = self.helical_model.accelerator.gather_for_metrics((embedding))
        #     if self.helical_model.accelerator.is_main_process:
        #         embeddings = embedding
        # else:
        #     embeddings = embedding
        output = self.fine_tuning_head(embeddings)
        return output