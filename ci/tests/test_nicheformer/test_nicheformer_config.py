import numpy as np
from pathlib import Path

from helical.models.nicheformer import NicheformerConfig
from helical.models.nicheformer.nicheformer_config import (
    _DEFAULT_MODEL_NAME,
    _MODEL_FILES,
)

_EXPECTED_FILES = set(_MODEL_FILES)
_HF_BASE_URL = f"https://huggingface.co/{_DEFAULT_MODEL_NAME}/resolve/main"


class TestNicheformerConfig:
    def test_default_batch_size(self):
        assert NicheformerConfig().config["batch_size"] == 32

    def test_default_device(self):
        assert NicheformerConfig().config["device"] == "cpu"

    def test_default_layer(self):
        assert NicheformerConfig().config["layer"] == -1

    def test_default_with_context(self):
        assert NicheformerConfig().config["with_context"] is False

    def test_default_technology_mean(self):
        assert NicheformerConfig().config["technology_mean"] is None

    def test_default_model_name(self):
        assert NicheformerConfig().config["model_name"] == "theislab/Nicheformer"

    def test_custom_model_name(self):
        config = NicheformerConfig(model_name="myorg/MyNicheformer")
        assert config.config["model_name"] == "myorg/MyNicheformer"

    def test_custom_batch_size(self):
        assert NicheformerConfig(batch_size=16).config["batch_size"] == 16

    def test_custom_device(self):
        assert NicheformerConfig(device="cuda").config["device"] == "cuda"

    def test_custom_layer(self):
        assert NicheformerConfig(layer=6).config["layer"] == 6

    def test_custom_with_context(self):
        assert NicheformerConfig(with_context=True).config["with_context"] is True

    def test_technology_mean_as_ndarray(self):
        arr = np.ones(100)
        config = NicheformerConfig(technology_mean=arr)
        assert config.config["technology_mean"] is arr

    def test_technology_mean_as_path_string(self):
        config = NicheformerConfig(technology_mean="path/to/mean.npy")
        assert config.config["technology_mean"] == "path/to/mean.npy"

    def test_files_to_download_count(self):
        config = NicheformerConfig()
        assert len(config.list_of_files_to_download) == len(_EXPECTED_FILES)

    def test_files_to_download_are_path_url_tuples(self):
        config = NicheformerConfig()
        for local_path, url in config.list_of_files_to_download:
            assert isinstance(local_path, Path)
            assert isinstance(url, str)

    def test_files_to_download_urls_point_to_hf(self):
        config = NicheformerConfig()
        for _, url in config.list_of_files_to_download:
            assert url.startswith(_HF_BASE_URL)

    def test_files_to_download_covers_all_expected_files(self):
        config = NicheformerConfig()
        downloaded_filenames = {
            Path(url).name for _, url in config.list_of_files_to_download
        }
        assert downloaded_filenames == _EXPECTED_FILES

    def test_local_paths_are_under_model_dir(self):
        config = NicheformerConfig()
        for local_path, _ in config.list_of_files_to_download:
            assert local_path.parent == config.model_dir
