"""Seed/device discipline smoke tests (local-only: needs torch, not in CI subset)."""
from __future__ import annotations

import numpy as np
import torch

from otraj.utils import resolve_device, set_seed


def test_set_seed_reproduces_streams() -> None:
    set_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_set_seed_changes_with_seed() -> None:
    set_seed(1)
    a = torch.rand(3)
    set_seed(2)
    b = torch.rand(3)
    assert not torch.equal(a, b)


def test_resolve_device_explicit_and_auto() -> None:
    assert resolve_device("cpu").type == "cpu"
    dev = resolve_device(None)
    assert dev.type in ("cuda", "cpu")
    if torch.cuda.is_available():
        assert dev.type == "cuda"
