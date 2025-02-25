from typing import Literal


class Evo2Config:
    """Configuration class for Evo 2 models.

    Parameters
    ----------
    model_name : str, optional
        The name of the model to use. The default is "evo2-7b". Options are "evo2-7b", "evo2-7b-base", "evo2-1b-base".
    batch_size : int, optional
        The batch size to use. The default is 1.
    """

    def __init__(
        self,
        model_name: Literal["evo2-7b", "evo2-7b-base", "evo2-1b-base"] = "evo2-7b",
        batch_size: int = 1,
    ):
        # model specific parameters
        self.model_map = {
            "evo2-7b": {
                "model_name": "evo2_7b",
                "model_hf_name": "arcinstitute/evo2_7b",
                "default_embedding_layer": "blocks.31.mlp.l3",
                "vocab_size": 512,
                "hidden_size": 4096,
                # Number of long convolution filters in each hyena block. Can be smaller than `hidden_size`
                "num_filters": 4096,
                "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23, 27, 30],
                "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22, 26, 29],
                "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21, 25, 28],
                "attn_layer_idxs": [3, 10, 17, 24, 31],
                "hcm_filter_length": 128,
                "hcl_filter_groups": 4096,
                "hcm_filter_groups": 256,
                "hcs_filter_groups": 256,
                "hcs_filter_length": 7,
                "num_layers": 32,
                # Length of the short, depthwise FIR applied to input projections
                "short_filter_length": 3,
                "num_attention_heads": 32,
                "short_filter_bias": False,  # add bias to FIR
                "mlp_init_method": "torch.nn.init.zeros_",
                "mlp_output_init_method": "torch.nn.init.zeros_",
                "eps": 0.000001,
                "state_size": 16,
                "rotary_emb_base": 100000000000,
                "rotary_emb_scaling_factor": 128,
                "use_interpolated_rotary_pos_emb": True,
                "make_vocab_size_divisible_by": 8,
                "inner_size_multiple_of": 16,  # force GLU inner_size to be a multiple of
                "inner_mlp_size": 11264,
                "log_intermediate_values": False,
                # Number of groups in GQA
                "proj_groups": 1,
                # Number of groups in grouped
                "hyena_filter_groups": 1,
                # Split strategy for channels
                "column_split_hyena": False,
                "column_split": True,
                "interleave": True,
                # Layer > 0 nn.identity activation
                "evo2_style_activations": True,
                # Legacy options for MP / PP inference
                "model_parallel_size": 1,
                "pipe_parallel_size": 1,
                "tie_embeddings": True,
                "mha_out_proj_bias": True,
                "hyena_out_proj_bias": True,
                "hyena_flip_x1x2": False,
                "qkv_proj_bias": False,
                "use_fp8_input_projections": True,
                "max_seqlen": 1048576,
                "final_norm": True,
                "use_flash_attn": True,
                "use_flash_rmsnorm": False,
                "use_flash_depthwise": False,
                "use_flashfft": False,
                "use_laughing_hyena": False,
                "inference_mode": True,
                "prefill_style": "fft",
                "mlp_activation": "gelu",
                "print_activations": False,
            },
            "evo2-7b-base": {
                "model_name": "evo2_7b_base",
                "model_hf_name": "arcinstitute/evo2_7b_base",
                "default_embedding_layer": "blocks.31.mlp.l3",
                "vocab_size": 512,
                "hidden_size": 1920,
                # Number of independent filters in Hyena-LI
                "num_filters": 1920,
                "attn_layer_idxs": [3, 10, 17, 24],
                "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23],
                "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22],
                "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21],
                "hcm_filter_length": 128,
                "hcl_filter_groups": 1920,
                "hcm_filter_groups": 128,
                "hcs_filter_groups": 128,
                "hcs_filter_length": 7,
                "num_layers": 25,
                "short_filter_length": 3,
                "num_attention_heads": 15,
                "short_filter_bias": False,  # add bias to FIR
                "mlp_init_method": "torch.nn.init.zeros_",
                "mlp_output_init_method": "torch.nn.init.zeros_",
                "eps": 0.000001,
                "state_size": 16,
                "rotary_emb_base": 10000,
                "make_vocab_size_divisible_by": 8,
                "inner_size_multiple_of": 16,  # force GLU inner_size to be a multiple of
                "inner_mlp_size": 5120,
                "log_intermediate_values": False,
                # Number of groups in GQA
                "proj_groups": 1,
                # Number of groups in grouped
                "hyena_filter_groups": 1,
                # Split strategy for channels
                "column_split_hyena": False,
                "column_split": True,
                "interleave": True,
                # Layer > 0 nn.identity activation
                "evo2_style_activations": True,
                # Legacy options for MP / PP inference
                "model_parallel_size": 1,
                "pipe_parallel_size": 1,
                "tie_embeddings": True,
                "mha_out_proj_bias": True,
                "hyena_out_proj_bias": True,
                "hyena_flip_x1x2": False,
                "qkv_proj_bias": False,
                "use_fp8_input_projections": True,
                "max_seqlen": 8192,
                "final_norm": True,
                "use_flash_attn": True,
                "use_flash_rmsnorm": False,
                "use_flash_depthwise": False,
                "use_flashfft": False,
                "use_laughing_hyena": False,
                "inference_mode": True,
                "prefill_style": "fft",
                "mlp_activation": "gelu",
                "print_activations": False,
            },
            "evo2-1b-base": {
                "model_name": "evo2_1b_base",
                "model_hf_name": "arcinstitute/evo2_1b_base",
                "default_embedding_layer": "blocks.24.mlp.l3",
                "vocab_size": 512,
                "hidden_size": 1920,
                # Number of independent filters in Hyena-LI
                "num_filters": 1920,
                "attn_layer_idxs": [3, 10, 17, 24],
                "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23],
                "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22],
                "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21],
                "hcm_filter_length": 128,
                "hcl_filter_groups": 1920,
                "hcm_filter_groups": 128,
                "hcs_filter_groups": 128,
                "hcs_filter_length": 7,
                "num_layers": 25,
                # Length of the short, depthwise FIR applied to input projections
                "short_filter_length": 3,
                "num_attention_heads": 15,
                "short_filter_bias": False,  # add bias to FIR
                "mlp_init_method": "torch.nn.init.zeros_",
                "mlp_output_init_method": "torch.nn.init.zeros_",
                "eps": 0.000001,
                "state_size": 16,
                "rotary_emb_base": 10000,
                "make_vocab_size_divisible_by": 8,
                "inner_size_multiple_of": 16,  # force GLU inner_size to be a multiple of
                "inner_mlp_size": 5120,
                "log_intermediate_values": False,
                # Number of groups in GQA
                "proj_groups": 1,
                # Number of groups in grouped
                "hyena_filter_groups": 1,
                # Split strategy for channels
                "column_split_hyena": False,
                "column_split": True,
                "interleave": True,
                # Layer > 0 nn.identity activation
                "evo2_style_activations": True,
                # Legacy options for MP / PP inference
                "model_parallel_size": 1,
                "pipe_parallel_size": 1,
                "tie_embeddings": True,
                "mha_out_proj_bias": True,
                "hyena_out_proj_bias": True,
                "hyena_flip_x1x2": False,
                "qkv_proj_bias": False,
                "use_fp8_input_projections": True,
                "max_seqlen": 8192,
                "final_norm": True,
                "use_flash_attn": True,
                "use_flash_rmsnorm": False,
                "use_flash_depthwise": False,
                "use_flashfft": False,
                "use_laughing_hyena": False,
                "inference_mode": True,
                "prefill_style": "fft",
                "mlp_activation": "gelu",
                "print_activations": False,
            },
        }
        if model_name not in self.model_map:
            raise ValueError(
                f"Model name {model_name} not found in available models: {self.model_map.keys()}"
            )

        self.config = {
            "model_map": self.model_map[model_name],
            "batch_size": batch_size,
            "device": "cuda",
        }
