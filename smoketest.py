import torch

def test_torch_imports():
    assert tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 4)

def test_small_op_cpu():
    x = torch.randn(64, 64)
    y = torch.mm(x, x.t())
    assert y.shape == (64, 64) and torch.isfinite(y).all()

def test_cuda_if_available():
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.t()
        assert y.is_cuda and torch.isfinite(y).all()

def test_mps_if_available():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
        x = torch.randn(512, 512, device="mps")
        y = x @ x.t()
        assert y.device.type == "mps"