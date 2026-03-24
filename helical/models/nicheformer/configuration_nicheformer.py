from transformers import PretrainedConfig


class NicheformerConfig(PretrainedConfig):
    model_type = "nicheformer"

    def __init__(
        self,
        dim_model=512,
        nheads=16,
        dim_feedforward=1024,
        nlayers=12,
        dropout=0.0,
        batch_first=True,
        masking_p=0.15,
        n_tokens=20340,
        context_length=1500,
        cls_classes=164,
        supervised_task=None,
        learnable_pe=True,
        specie=True,
        assay=True,
        modality=True,
        **kwargs,
    ):
        """Initialize NicheformerConfig.

        Args:
            dim_model: Dimensionality of the model
            nheads: Number of attention heads
            dim_feedforward: Dimensionality of MLPs in attention blocks
            nlayers: Number of transformer layers
            dropout: Dropout probability
            batch_first: Whether batch dimension is first
            masking_p: Probability of masking tokens
            n_tokens: Total number of tokens (excluding auxiliary)
            context_length: Length of the context window
            cls_classes: Number of classification classes
            supervised_task: Type of supervised task
            learnable_pe: Whether to use learnable positional embeddings
            specie: Whether to add specie token
            assay: Whether to add assay token
            modality: Whether to add modality token
        """
        super().__init__(**kwargs)

        self.dim_model = dim_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.nlayers = nlayers
        self.dropout = dropout
        self.batch_first = batch_first
        self.masking_p = masking_p
        self.n_tokens = n_tokens
        self.context_length = context_length
        self.cls_classes = cls_classes
        self.supervised_task = supervised_task
        self.learnable_pe = learnable_pe
        self.specie = specie
        self.assay = assay
        self.modality = modality
