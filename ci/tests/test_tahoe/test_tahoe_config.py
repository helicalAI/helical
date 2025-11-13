import pytest
from helical.models.tahoe import TahoeConfig


class TestTahoeConfig:
    """Test suite for TahoeConfig class."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = TahoeConfig()

        assert config.config["model_size"] == "70m"
        assert config.config["batch_size"] == 10
        assert config.config["emb_mode"] == "cell"
        assert config.config["attn_impl"] == "flash"
        assert config.config["device"] == "cuda"
        assert config.config["max_length"] == 10000
        assert config.config["num_workers"] == 8
        assert config.config["prefetch_factor"] == 48

    def test_custom_model_size(self):
        """Test configuration with custom model size."""
        config = TahoeConfig(model_size="1b")
        assert config.config["model_size"] == "1b"

    def test_custom_batch_size(self):
        """Test configuration with custom batch size."""
        config = TahoeConfig(batch_size=32)
        assert config.config["batch_size"] == 32

    def test_custom_emb_mode(self):
        """Test configuration with custom embedding mode."""
        config = TahoeConfig(emb_mode="gene")
        assert config.config["emb_mode"] == "gene"

    def test_custom_attn_impl(self):
        """Test configuration with custom attention implementation."""
        config = TahoeConfig(attn_impl="torch")
        assert config.config["attn_impl"] == "torch"

    def test_custom_device(self):
        """Test configuration with custom device."""
        config = TahoeConfig(device="cpu")
        assert config.config["device"] == "cpu"

    def test_custom_max_length(self):
        """Test configuration with custom max length."""
        config = TahoeConfig(max_length=5000)
        assert config.config["max_length"] == 5000

    def test_multiple_custom_parameters(self):
        """Test configuration with multiple custom parameters."""
        config = TahoeConfig(
            model_size="1b",
            batch_size=16,
            emb_mode="gene",
            attn_impl="torch",
            device="cpu",
            max_length=8000,
            num_workers=4
        )

        assert config.config["model_size"] == "1b"
        assert config.config["batch_size"] == 16
        assert config.config["emb_mode"] == "gene"
        assert config.config["attn_impl"] == "torch"
        assert config.config["device"] == "cpu"
        assert config.config["max_length"] == 8000
        assert config.config["num_workers"] == 4

    @pytest.mark.parametrize("emb_mode", ["cell", "gene"])
    def test_valid_emb_modes(self, emb_mode):
        """Test that valid embedding modes are accepted."""
        config = TahoeConfig(emb_mode=emb_mode)
        assert config.config["emb_mode"] == emb_mode

    @pytest.mark.parametrize("attn_impl", ["flash", "torch", "triton"])
    def test_valid_attn_impl(self, attn_impl):
        """Test that valid attention implementations are accepted."""
        config = TahoeConfig(attn_impl=attn_impl)
        assert config.config["attn_impl"] == attn_impl

    def test_hf_repo_id(self):
        """Test that HuggingFace repository ID is set correctly."""
        config = TahoeConfig()
        assert config.config["hf_repo_id"] == "tahoebio/Tahoe-x1"

    def test_config_immutability(self):
        """Test that config can be modified after creation."""
        config = TahoeConfig(batch_size=10)
        config.config["batch_size"] = 20
        assert config.config["batch_size"] == 20
