from omegaconf import DictConfig

class TranscriptFormerConfig:
    def __init__(self,
        config: DictConfig = None,
    ):
        self.config = config

        self.list_of_files_to_download = [
            "transcriptformer/tf_sapiens/config.json",
            "transcriptformer/tf_sapiens/model_weights.pt",
            "transcriptformer/tf_sapiens/vocabs/assay_vocab.json",
            "transcriptformer/tf_sapiens/vocabs/homo_sapiens_gene.h5",
        ]
