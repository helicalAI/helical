from typing import Optional
class scGPTConfig():
    """
    Configuration class to use the scGPT Model.
    
    Parameters
    ----------
    data_source : str, optional, default = "/scratch/ssd004/datasets/cellxgene/scb_strict/human"
        The data source
    save_dir : str, optional, default = "/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-Dec18-13-52-2023"
        The save directory
    load_model : str, optional, default = "/scratch/ssd004/datasets/cellxgene/save/scGPT_human"
        The model to load
    n_hvg : int, optional, default = None
        The number of highly variable genes
    valid_size_or_ratio : float, optional, default = 0.003
        The validation size or ratio
    dist_backend : str, optional, default = "nccl"
        The distributed backend
    grad_accu_steps : int, optional, default = 1
        The gradient accumulation steps
    pad_token : str, optional, default = "<pad>"
        The padding token
    input_style : str, optional, default = "binned"
        The input style
    input_emb_style : str, optional, default = "continuous"
        The input embedding style
    n_bins : int, optional, default = 51
        The number of bins
    max_seq_len : int, optional, default = 1200
        The maximum sequence length
    training_tasks : str, optional, default = "both"
        The training tasks
    dist_url : str, optional, default = "tcp://gpu183.cluster.local:54165"
        The distributed URL
    mask_ratio : list, optional, default = [0.25,0.5,0.75]
        The mask ratio
    trunc_by_sample : bool, optional, default = True   
        Whether to truncate by sample
    vocab_path : str, optional, default = "/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json"
        The vocabulary path
    rank : int, optional, default = 0
        The rank
    batch_size : int, optional, default = 24
        The batch size
    eval_batch_size : int, optional, default = 48
        The evaluation batch size
    epochs : int, optional, default = 10
        The number of epochs
    lr : float, optional, default = 0.0001
        The learning rate
    scheduler_interval : int, optional, default = 100
        The scheduler interval
    scheduler_factor : float, optional, default = 0.99
        The scheduler factor
    warmup_ratio_or_step : float, optional, default = 10000.0
        The warmup ratio or step
    no_cls : bool, optional, default = False
        Whether to use the classification token
    no_cce : bool, optional, default = True
        Whether to use the cross-entropy loss
    fp16 : bool, optional, default = True
        Whether to use fp16
    fast_transformer : bool, optional, default = True
        Whether to use the fast transformer
    annotation_source : str, optional, default = "/scratch/ssd004/datasets/cellxgene/tabula_sapiens/parquet/"
        The annotation source
    annotation_valid_size_or_ratio : float, optional, default = 0.1
        The annotation validation size or ratio
    nlayers : int, optional, default = 12
        The number of layers
    nheads : int, optional, default = 8
        The number of heads
    embsize : int, optional, default = 512
        The embedding size
    d_hid : int, optional, default = 512
        The hidden dimension
    dropout : float, optional, default = 0.2
        The dropout
    n_layers_cls : int, optional, default = 3
        The number of classification layers
    annote_max_seq_len : int, optional, default = 5000
        The annotation maximum sequence length
    log_interval : int, optional, default = 500
        The logging interval
    save_interval : int, optional, default = 1000
        The save interval
    mask_value : int, optional, default = -1
        The mask value
    pad_value : int, optional, default = -2
        The padding value
    USE_CLS : bool, optional, default = True
        Whether to use the classification token
    USE_CCE : bool, optional, default = False
        Whether to use the cross-entropy loss
    MVC : bool, optional, default = True
        Whether to use the MVC
    USE_GENERATIVE_TRAINING : bool, optional, default = True   
        Whether to use generative training
    world_size : int, optional, default = 8
        The world size
    distributed : bool, optional, default = True
        Whether to use distributed training
    local_rank : int, optional, default = 0
        The local rank
    gpu : int, optional, default = 0
        The GPU
    accelerator : dict, optional, default = None
        The accelerator configuration

    Returns
    -------
    scGPTConfig 
       The scGPT configuration object

    Notes
    -----
    This configuration contains all the dwfault parameteres that have been used in the original scGPT repository.

    """

    def __init__(
            self, 
            data_source: str = "/scratch/ssd004/datasets/cellxgene/scb_strict/human",
            save_dir: str = "/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-Dec18-13-52-2023",
            load_model: str = "/scratch/ssd004/datasets/cellxgene/save/scGPT_human",
            n_hvg: Optional[int] = None,
            valid_size_or_ratio: float = 0.003,
            dist_backend: str = "nccl",
            grad_accu_steps: int = 1,
            pad_token: str = "<pad>",
            input_style: str = "binned",
            input_emb_style: str = "continuous",
            n_bins: int = 51,
            max_seq_len: int = 1200,
            training_tasks: str = "both",
            dist_url: str = "tcp://gpu183.cluster.local:54165",
            mask_ratio: list = [0.25,0.5,0.75],
            trunc_by_sample: bool = True,
            vocab_path: str = "/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json",
            rank: int = 0,           
            batch_size: int = 24,
            eval_batch_size: int = 48,
            epochs: int = 10,
            lr: float = 0.0001,
            scheduler_interval: int = 100,
            scheduler_factor: float = 0.99,
            warmup_ratio_or_step: float = 10000.0,
            no_cls: bool = False,
            no_cce: bool = True,
            fp16: bool = True,
            fast_transformer: bool = True,
            annotation_source: str = "/scratch/ssd004/datasets/cellxgene/tabula_sapiens/parquet/",
            annotation_valid_size_or_ratio: float = 0.1,
            nlayers: int = 12,
            nheads: int = 8,
            embsize: int = 512,
            d_hid: int = 512,
            dropout: float = 0.2,
            n_layers_cls: int = 3,
            annote_max_seq_len: int = 5000,
            log_interval: int = 500,
            save_interval: int = 1000,
            mask_value: int = -1,
            pad_value: int = -2,
            USE_CLS: bool = True,
            USE_CCE:  bool = False,
            MVC: bool = True,
            USE_GENERATIVE_TRAINING: bool = True,
            world_size: int = 8,
            distributed: bool = True,
            local_rank: int = 0,
            gpu: int = 0,
            accelerator: Optional[dict] = None
            ):

        self.config = {
            "data_source": data_source,
            "save_dir": save_dir,
            "load_model": load_model,
            "n_hvg": n_hvg,
            "valid_size_or_ratio": valid_size_or_ratio,
            "dist_backend": dist_backend,
            "grad_accu_steps": grad_accu_steps,
            "pad_token": pad_token,
            "input_style": input_style,
            "input_emb_style": input_emb_style,
            "n_bins": n_bins,
            "max_seq_len": max_seq_len,
            "training_tasks": training_tasks,
            "dist_url": dist_url,
            "mask_ratio": mask_ratio,
            "trunc_by_sample": trunc_by_sample,
            "vocab_path": vocab_path,
            "rank": rank,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": epochs,
            "lr": lr,
            "scheduler_interval": scheduler_interval,
            "scheduler_factor": scheduler_factor,
            "warmup_ratio_or_step": warmup_ratio_or_step,
            "no_cls": no_cls,
            "no_cce": no_cce,
            "fp16": fp16,
            "fast_transformer": fast_transformer,
            "annotation_source": annotation_source,
            "annotation_valid_size_or_ratio": annotation_valid_size_or_ratio,
            "nlayers": nlayers,
            "nheads": nheads,
            "embsize": embsize,
            "d_hid": d_hid,
            "dropout": dropout,
            "n_layers_cls": n_layers_cls,
            "annote_max_seq_len": annote_max_seq_len,
            "log_interval": log_interval,
            "save_interval": save_interval,
            "mask_value": mask_value,
            "pad_value": pad_value,
            "USE_CLS": USE_CLS,
            "USE_CCE": USE_CCE,
            "MVC": MVC,
            "USE_GENERATIVE_TRAINING": USE_GENERATIVE_TRAINING,
            "world_size": world_size,
            "distributed": distributed,
            "local_rank": local_rank,
            "gpu" : gpu,
            "accelerator": accelerator
            }
        