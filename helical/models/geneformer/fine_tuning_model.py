from helical.models.base_models import HelicalBaseFineTuningHead
import torch

class GeneformerFineTuningModel(torch.nn.Module):
    """Fine-tuning model for the Geneformer model.
    
    Parameters
    ----------
    helical_model : Geneformer
        The initialised Geneformer model to fine-tune.
    fine_tuning_head : HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model.
    
    Methods
    -------
    forward(input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor
        The forward method of the fine-tuning model.
    """
    def __init__(self, geneformer_model, fine_tuning_head: HelicalBaseFineTuningHead):
        super(GeneformerFineTuningModel, self).__init__()
        self.helical_model = geneformer_model
        self.fine_tuning_head = fine_tuning_head

    def forward(self, input_ids: torch.Tensor, attention_mask_minibatch: torch.Tensor) -> torch.Tensor:
        outputs = self.helical_model.forward(input_ids=input_ids, attention_mask=attention_mask_minibatch)
        final_layer = outputs.hidden_states[-1]
        cls_seq = final_layer[:, 0, :]
        final = self.fine_tuning_head(cls_seq)
        return final