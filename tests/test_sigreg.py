"""Tests for SIGReg implementation."""

import torch
import pytest
from models.ssl.sigreg import SIGReg, _EppsPulleyFallback


def test_sigreg_scalar_output():
    sigreg = SIGReg(embed_dim=64, num_proj=32)
    z = torch.randn(16, 64)
    loss = sigreg(z)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"


def test_sigreg_no_nan():
    sigreg = SIGReg(embed_dim=64, num_proj=32)
    z = torch.randn(16, 64)
    loss = sigreg(z)
    assert not loss.isnan(), "SIGReg returned NaN"
    assert not loss.isinf(), "SIGReg returned Inf"


def test_sigreg_collapse_detection():
    """Collapsed representations should produce high SIGReg loss."""
    sigreg = SIGReg(embed_dim=64, num_proj=64)
    # Normal representations
    z_good = torch.randn(32, 64)
    loss_good = sigreg(z_good).item()
    # Collapsed representations (all identical)
    z_collapsed = torch.zeros(32, 64) + 0.01
    loss_collapsed = sigreg(z_collapsed).item()
    assert loss_collapsed > loss_good, (
        f"Collapsed loss ({loss_collapsed:.4f}) should be > normal ({loss_good:.4f})"
    )


def test_sigreg_bf16_input():
    """SIGReg must handle bf16 input without NaN."""
    sigreg = SIGReg(embed_dim=64, num_proj=32)
    z = torch.randn(8, 64).to(torch.bfloat16)
    loss = sigreg(z)
    assert not loss.isnan()


def test_sigreg_gradient_flows():
    """Gradients must flow through the encoder embeddings."""
    sigreg = SIGReg(embed_dim=32, num_proj=16)
    # Simulate encoder output requiring gradient
    linear = torch.nn.Linear(32, 32)
    z_input = torch.randn(8, 32)
    z = linear(z_input)
    loss = sigreg(z)
    loss.backward()
    assert linear.weight.grad is not None, "No gradient through SIGReg"


def test_epps_pulley_shape():
    ep = _EppsPulleyFallback(t_max=3.0, n_points=17)
    x = torch.randn(32, 64)  # (N_samples, N_proj)
    stat = ep(x)
    assert stat.shape == (64,), f"Expected (64,), got {stat.shape}"


def test_sigreg_batch_dims():
    """SIGReg should handle (B*N, D) input from token flattening."""
    sigreg = SIGReg(embed_dim=384, num_proj=32)
    z = torch.randn(4 * 512, 384)  # B=4, N_tokens=512
    loss = sigreg(z)
    assert loss.shape == ()
