from typing import Optional
class GeneformerConfig():
    """Configuration class to use the Geneformer Model.
    
    Parameters
    ----------
    model_name : str, optional, default = "geneformer-12L-30M"
        The model name
    batch_size : int, optional, default = 5
        The batch size
    emb_layer : int, optional, default = -1
        The embedding layer
    emb_mode : str, optional, default = "cell"
        The embedding mode
    device : str, optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu"
    accelerator : dict, optional, default = {"cpu": True}
        The accelerator configuration

    Returns
    -------
    GeneformerConfig
        The Geneformer configuration object

    """
    def __init__(
            self, 
            model_name: str = "geneformer-12L-30M",
            batch_size: int = 5,
            emb_layer: int = -1,
            emb_mode: str = "cell",
            device: str = "cpu",
            accelerator: Optional[dict] = {"cpu": True}
            ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.emb_layer = emb_layer
        self.emb_mode = emb_mode
        self.device = device
        self.accelerator = accelerator


