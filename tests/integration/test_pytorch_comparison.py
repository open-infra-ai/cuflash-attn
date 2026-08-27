#!/usr/bin/env python3
from __future__ import annotations

"""
PyTorch Comparison Tests for cuflash
Feature: cuflash
Validates: Requirements 8.4

This script compares the cuflash implementation against PyTorch's
scaled_dot_product_attention for numerical correctness by invoking the
compiled library directly through a small C ABI wrapper.
"""

import ctypes
import os
import sys
from typing import Optional, Sequence, Tuple

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as import_error:
    np = None
    torch = None
    F = None
    DEPENDENCY_IMPORT_ERROR = import_error
else:
    DEPENDENCY_IMPORT_ERROR = None


SUCCESS = 0
UNSUPPORTED_DTYPE = 7
SKIP = 77


class CuFlashLibrary:
    def __init__(self, library: ctypes.CDLL):
        self.library = library

        self.library.cuflash_attention_forward_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_forward_f32.restype = ctypes.c_int

        self.library.cuflash_attention_backward_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_backward_f32.restype = ctypes.c_int

        self.library.cuflash_attention_forward_f16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_forward_f16.restype = ctypes.c_int

        self.library.cuflash_attention_backward_f16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_backward_f16.restype = ctypes.c_int

        self.library.cuflash_attention_forward_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_forward_bf16.restype = ctypes.c_int

        self.library.cuflash_attention_backward_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.cuflash_attention_backward_bf16.restype = ctypes.c_int


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_script_dir(), "..", ".."))


def _candidate_library_paths() -> Sequence[str]:
    root = _repo_root()
    build_dir = os.path.join(root, "build")
    candidates = []

    explicit = os.environ.get("CUFLASH_LIB")
    if explicit:
        candidates.append(explicit)

    if sys.platform.startswith("win"):
        names = ["cuflash.dll", "libcuflash.dll"]
    elif sys.platform == "darwin":
        names = ["libcuflash.dylib"]
    else:
        names = ["libcuflash.so"]

    preset_dirs = ["default", "release", "release-fast-math", "minimal", "Release", "Debug"]
    for name in names:
        candidates.append(os.path.join(build_dir, name))
        for preset_dir in preset_dirs:
            candidates.append(os.path.join(build_dir, preset_dir, name))

    return candidates


def load_library() -> Optional[CuFlashLibrary]:
    for lib_path in _candidate_library_paths():
        try:
            return CuFlashLibrary(ctypes.CDLL(lib_path))
        except OSError:
            continue

    print("Library not found. Tried:")
    for lib_path in _candidate_library_paths():
        print(f"  - {lib_path}")
    print("Please build the project first with BUILD_SHARED_LIBS=ON.")
    return None


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    assert tensor.is_cuda, "Tensor must be on CUDA"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    return ctypes.c_void_p(tensor.data_ptr())


def _call_forward_f32(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    o = torch.empty_like(q)
    l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    status = library.library.cuflash_attention_forward_f32(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, o, l


def _call_backward_f32(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    l: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)

    status = library.library.cuflash_attention_backward_f32(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        _ptr(grad_output),
        _ptr(d_q),
        _ptr(d_k),
        _ptr(d_v),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, d_q, d_k, d_v


def _call_forward_f16(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    o = torch.empty_like(q)
    # logsumexp is stored in FP32 (the C ABI takes a float* even for FP16
    # inputs) so the backward can reconstruct softmax probabilities accurately.
    l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    status = library.library.cuflash_attention_forward_f16(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, o, l


def _call_backward_f16(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    l: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)

    status = library.library.cuflash_attention_backward_f16(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        _ptr(grad_output),
        _ptr(d_q),
        _ptr(d_k),
        _ptr(d_v),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, d_q, d_k, d_v


def _call_forward_bf16(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    o = torch.empty_like(q)
    # logsumexp is stored in FP32 (the C ABI takes a float* even for BF16
    # inputs) so the backward can reconstruct softmax probabilities accurately.
    l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    status = library.library.cuflash_attention_forward_bf16(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, o, l


def _call_backward_bf16(
    library: CuFlashLibrary,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    l: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = float(1.0 / np.sqrt(head_dim))
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)

    status = library.library.cuflash_attention_backward_bf16(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(o),
        _ptr(l),
        _ptr(grad_output),
        _ptr(d_q),
        _ptr(d_k),
        _ptr(d_v),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        ctypes.c_float(scale),
        causal,
        None,
    )
    torch.cuda.synchronize()
    return status, d_q, d_k, d_v


def _bf16_native() -> bool:
    # Native bfloat16 tensor-core support starts with Ampere (sm_80).
    return torch.cuda.get_device_capability(0) >= (8, 0)


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scale = float(1.0 / np.sqrt(q.shape[-1]))
    return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)


def test_forward_equivalence(library: CuFlashLibrary):
    """Test that FP32 forward pass matches PyTorch."""
    print("Testing forward pass equivalence...")

    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 64

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()

    status, output, _ = _call_forward_f32(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash FP32 forward failed with status {status}"

    reference = _reference_attention(q, k, v, causal=False)
    diff = (output - reference).abs().max().item()
    print(f"  CuFlash vs PyTorch forward max diff: {diff:.2e}")
    assert diff < 1e-3, f"Forward equivalence test failed: {diff}"

    print("  Forward pass test PASSED")
    return True


def test_causal_mask(library: CuFlashLibrary):
    """Test that causal FP32 forward pass matches PyTorch."""
    print("Testing causal mask...")

    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()

    status, output, _ = _call_forward_f32(library, q, k, v, causal=True)
    assert status == SUCCESS, f"CuFlash causal forward failed with status {status}"

    reference = _reference_attention(q, k, v, causal=True)
    diff = (output - reference).abs().max().item()
    print(f"  CuFlash vs PyTorch causal max diff: {diff:.2e}")
    assert diff < 1e-3, f"Causal mask test failed: {diff}"

    print("  Causal mask test PASSED")
    return True


def test_backward_gradients(library: CuFlashLibrary):
    """Test that FP32 backward gradients match PyTorch autograd."""
    print("Testing backward pass gradients...")

    batch_size = 1
    num_heads = 2
    seq_len = 16
    head_dim = 32

    torch.manual_seed(42)
    q_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)

    grad_output = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()

    reference_output = _reference_attention(q_ref, k_ref, v_ref, causal=False)
    reference_output.backward(grad_output)

    q = q_ref.detach().contiguous()
    k = k_ref.detach().contiguous()
    v = v_ref.detach().contiguous()

    status, output, l = _call_forward_f32(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash forward before backward failed with status {status}"

    status, d_q, d_k, d_v = _call_backward_f32(library, q, k, v, output, l, grad_output, causal=False)
    assert status == SUCCESS, f"CuFlash backward failed with status {status}"

    d_q_diff = (d_q - q_ref.grad).abs().max().item()
    d_k_diff = (d_k - k_ref.grad).abs().max().item()
    d_v_diff = (d_v - v_ref.grad).abs().max().item()

    print(f"  dQ max diff: {d_q_diff:.2e}")
    print(f"  dK max diff: {d_k_diff:.2e}")
    print(f"  dV max diff: {d_v_diff:.2e}")

    assert d_q_diff < 1e-3, f"dQ mismatch: {d_q_diff}"
    assert d_k_diff < 1e-3, f"dK mismatch: {d_k_diff}"
    assert d_v_diff < 1e-3, f"dV mismatch: {d_v_diff}"

    print("  Backward pass test PASSED")
    return True


def test_fp16_forward_equivalence(library: CuFlashLibrary):
    """Test that FP16 forward pass stays close to PyTorch."""
    print("Testing FP16 forward equivalence...")

    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()

    status, output, _ = _call_forward_f16(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash FP16 forward failed with status {status}"

    reference = _reference_attention(q, k, v, causal=False)
    diff = (output.float() - reference.float()).abs().max().item()
    print(f"  CuFlash FP16 vs PyTorch max diff: {diff:.2e}")
    assert diff < 1e-2, f"FP16 forward equivalence test failed: {diff}"

    print("  FP16 forward pass test PASSED")
    return True


def test_fp16_backward_equivalence(library: CuFlashLibrary):
    """Test that FP16 backward gradients stay close to PyTorch autograd."""
    print("Testing FP16 backward equivalence...")

    batch_size = 1
    num_heads = 1
    seq_len = 16
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    grad_output = torch.randn_like(q).contiguous()

    # CuFlash FP16 forward + backward
    status, output, l = _call_forward_f16(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash FP16 forward failed with status {status}"

    status, d_q, d_k, d_v = _call_backward_f16(library, q, k, v, output, l, grad_output, causal=False)
    assert status == SUCCESS, f"CuFlash FP16 backward failed with status {status}"

    # PyTorch reference (compute in float32 for numerical stability)
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    ref_output = _reference_attention(q_ref, k_ref, v_ref, causal=False)
    ref_output.backward(grad_output.float())

    # Compare with relaxed tolerance for FP16
    d_q_diff = (d_q.float() - q_ref.grad).abs().max().item()
    d_k_diff = (d_k.float() - k_ref.grad).abs().max().item()
    d_v_diff = (d_v.float() - v_ref.grad).abs().max().item()

    print(f"  dQ max diff: {d_q_diff:.2e}")
    print(f"  dK max diff: {d_k_diff:.2e}")
    print(f"  dV max diff: {d_v_diff:.2e}")

    # FP16 gradient error vs a float32 reference stays around 1e-2 for these
    # small shapes; 2e-2 leaves headroom for half-precision accumulation while
    # still catching real bugs (the previous 1e-1 was too loose to be useful).
    assert d_q_diff < 2e-2, f"dQ mismatch: {d_q_diff}"
    assert d_k_diff < 2e-2, f"dK mismatch: {d_k_diff}"
    assert d_v_diff < 2e-2, f"dV mismatch: {d_v_diff}"

    print("  FP16 backward equivalence test PASSED")
    return True


def test_fp16_causal_forward(library: CuFlashLibrary):
    """Test that causal FP16 forward pass stays close to PyTorch."""
    print("Testing FP16 causal forward...")

    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16).contiguous()

    status, output, _ = _call_forward_f16(library, q, k, v, causal=True)
    assert status == SUCCESS, f"CuFlash FP16 causal forward failed with status {status}"

    reference = _reference_attention(q, k, v, causal=True)
    diff = (output.float() - reference.float()).abs().max().item()
    print(f"  CuFlash FP16 vs PyTorch causal max diff: {diff:.2e}")
    assert diff < 1e-2, f"FP16 causal forward test failed: {diff}"

    print("  FP16 causal forward test PASSED")
    return True


def test_bf16_forward_equivalence(library: CuFlashLibrary):
    """Test that BF16 forward pass stays close to PyTorch."""
    if not _bf16_native():
        print("Testing BF16 forward equivalence... SKIPPED (requires sm_80+)")
        return True
    print("Testing BF16 forward equivalence...")

    batch_size = 1
    num_heads = 2
    seq_len = 32
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()

    status, output, _ = _call_forward_bf16(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash BF16 forward failed with status {status}"

    reference = _reference_attention(q, k, v, causal=False)
    diff = (output.float() - reference.float()).abs().max().item()
    print(f"  CuFlash BF16 vs PyTorch max diff: {diff:.2e}")
    # BF16 mantissa is coarser than FP16; same tolerance as test_dtype.cu.
    assert diff < 2e-2, f"BF16 forward equivalence test failed: {diff}"

    print("  BF16 forward pass test PASSED")
    return True


def test_bf16_backward_equivalence(library: CuFlashLibrary):
    """Test that BF16 backward gradients stay close to PyTorch autograd."""
    if not _bf16_native():
        print("Testing BF16 backward equivalence... SKIPPED (requires sm_80+)")
        return True
    print("Testing BF16 backward equivalence...")

    batch_size = 1
    num_heads = 1
    seq_len = 16
    head_dim = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    grad_output = torch.randn_like(q).contiguous()

    status, output, l = _call_forward_bf16(library, q, k, v, causal=False)
    assert status == SUCCESS, f"CuFlash BF16 forward failed with status {status}"

    status, d_q, d_k, d_v = _call_backward_bf16(library, q, k, v, output, l, grad_output, causal=False)
    assert status == SUCCESS, f"CuFlash BF16 backward failed with status {status}"

    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    ref_output = _reference_attention(q_ref, k_ref, v_ref, causal=False)
    ref_output.backward(grad_output.float())

    d_q_diff = (d_q.float() - q_ref.grad).abs().max().item()
    d_k_diff = (d_k.float() - k_ref.grad).abs().max().item()
    d_v_diff = (d_v.float() - v_ref.grad).abs().max().item()

    print(f"  dQ max diff: {d_q_diff:.2e}")
    print(f"  dK max diff: {d_k_diff:.2e}")
    print(f"  dV max diff: {d_v_diff:.2e}")

    assert d_q_diff < 2e-2, f"dQ mismatch: {d_q_diff}"
    assert d_k_diff < 2e-2, f"dK mismatch: {d_k_diff}"
    assert d_v_diff < 2e-2, f"dV mismatch: {d_v_diff}"

    print("  BF16 backward equivalence test PASSED")
    return True


def test_different_shapes(library: CuFlashLibrary):
    """Test various supported input shapes against PyTorch."""
    print("Testing different shapes...")

    test_cases = [
        (1, 1, 8, 32),
        (2, 4, 64, 64),
        (1, 8, 128, 64),
        (4, 2, 32, 128),
    ]

    for batch_size, num_heads, seq_len, head_dim in test_cases:
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float32).contiguous()

        status, output, _ = _call_forward_f32(library, q, k, v, causal=False)
        assert status == SUCCESS, f"CuFlash forward failed for shape {(batch_size, num_heads, seq_len, head_dim)} with status {status}"

        reference = _reference_attention(q, k, v, causal=False)
        diff = (output - reference).abs().max().item()

        assert output.shape == (batch_size, num_heads, seq_len, head_dim), (
            f"Output shape mismatch for input shape ({batch_size}, {num_heads}, {seq_len}, {head_dim})"
        )
        assert torch.isfinite(output).all(), (
            f"Output contains non-finite values for shape ({batch_size}, {num_heads}, {seq_len}, {head_dim})"
        )
        assert diff < 1e-3, (
            f"Output mismatch for shape ({batch_size}, {num_heads}, {seq_len}, {head_dim}): {diff}"
        )

        print(f"  Shape ({batch_size}, {num_heads}, {seq_len}, {head_dim}) PASSED")

    print("  Different shapes test PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("cuflash PyTorch Comparison Tests")
    print("=" * 60)

    if DEPENDENCY_IMPORT_ERROR is not None:
        print(f"Optional Python dependency not available, skipping tests: {DEPENDENCY_IMPORT_ERROR}")
        return SKIP

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return SKIP

    library = load_library()
    if library is None:
        return SKIP

    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        test_forward_equivalence,
        test_causal_mask,
        test_backward_gradients,
        test_fp16_forward_equivalence,
        test_fp16_backward_equivalence,
        test_fp16_causal_forward,
        test_bf16_forward_equivalence,
        test_bf16_backward_equivalence,
        test_different_shapes,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test(library):
                passed += 1
        except Exception as error:
            print(f"  FAILED: {error}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
