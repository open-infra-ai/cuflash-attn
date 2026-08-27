"""
Python binding example for cuflash using ctypes.

This example demonstrates how to call the cuflash C API from Python
without requiring PyTorch as a dependency.

Requirements:
    - CUDA-capable GPU
    - Built libcuflash.so library
    - numpy
    - cupy (optional, for GPU array management)

Usage:
    python examples/python_binding.py
"""

import ctypes
import sys
from ctypes import util
from pathlib import Path

import numpy as np


def find_library():
    """Find the cuflash shared library."""
    # Try common locations
    possible_paths = [
        Path(__file__).parent.parent / "build" / "release" / "libcuflash.so",
        Path(__file__).parent.parent / "build" / "debug" / "libcuflash.so",
        Path("/usr/local/lib/libcuflash.so"),
        Path("/usr/lib/libcuflash.so"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Try system library path
    try:
        return util.find_library("cuflash")
    except Exception:
        pass
    
    raise RuntimeError(
        "Could not find libcuflash.so. "
        "Please build the project first: cmake --preset release && cmake --build --preset release"
    )


def load_library(lib_path=None):
    """Load the cuflash shared library and set up function signatures."""
    if lib_path is None:
        lib_path = find_library()
    
    lib = ctypes.CDLL(lib_path)

    # Setup cuflash_attention_forward_f32
    lib.cuflash_attention_forward_f32.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # O (output)
        ctypes.c_void_p,  # L (softmax stats)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # num_heads
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # head_dim
        ctypes.c_float,   # scale
        ctypes.c_bool,    # causal
        ctypes.c_void_p,  # stream (cudaStream_t, pass None for default stream)
    ]
    lib.cuflash_attention_forward_f32.restype = ctypes.c_int

    # Setup cuflash_attention_backward_f32
    lib.cuflash_attention_backward_f32.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # O
        ctypes.c_void_p,  # L
        ctypes.c_void_p,  # dO (output gradient)
        ctypes.c_void_p,  # dQ (Q gradient)
        ctypes.c_void_p,  # dK (K gradient)
        ctypes.c_void_p,  # dV (V gradient)
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # num_heads
        ctypes.c_int,     # seq_len
        ctypes.c_int,     # head_dim
        ctypes.c_float,   # scale
        ctypes.c_bool,    # causal
        ctypes.c_void_p,  # stream (cudaStream_t, pass None for default stream)
    ]
    lib.cuflash_attention_backward_f32.restype = ctypes.c_int

    # Setup error string function
    lib.cuflash_error_string.argtypes = [ctypes.c_int]
    lib.cuflash_error_string.restype = ctypes.c_char_p

    return lib


class CuFlashAttn:
    """Python wrapper for cuflash library."""
    
    def __init__(self, lib_path=None):
        self.lib = load_library(lib_path)
    
    def check_error(self, error_code):
        """Check error code and raise exception if needed."""
        if error_code != 0:
            error_msg = self.lib.cuflash_error_string(error_code)
            raise RuntimeError(f"cuflash error {error_code}: {error_msg.decode('utf-8')}")
    
    def forward(self, Q, K, V, causal=False, scale=None):
        """
        Forward pass of FlashAttention.
        
        Args:
            Q: Query tensor (on GPU), shape (B, H, N, D)
            K: Key tensor (on GPU), shape (B, H, N, D)
            V: Value tensor (on GPU), shape (B, H, N, D)
            causal: Whether to use causal masking
            scale: Attention scale factor (default: 1/sqrt(D))
        
        Returns:
            O: Output tensor (on GPU), shape (B, H, N, D)
            L: Softmax statistics (on GPU), shape (B, H, N)
        """
        try:
            import cupy as cp
            use_cupy = True
        except ImportError:
            use_cupy = False
        
        B, H, N, D = Q.shape
        
        if scale is None:
            scale = 1.0 / np.sqrt(D)
        
        # Allocate output arrays
        if use_cupy:
            O = cp.empty_like(Q)
            L = cp.empty((B, H, N), dtype=np.float32)

            result = self.lib.cuflash_attention_forward_f32(
                ctypes.c_void_p(Q.data.ptr),
                ctypes.c_void_p(K.data.ptr),
                ctypes.c_void_p(V.data.ptr),
                ctypes.c_void_p(O.data.ptr),
                ctypes.c_void_p(L.data.ptr),
                B, H, N, D,
                ctypes.c_float(scale),
                causal,
                None  # stream: None = default CUDA stream
            )
        else:
            # Use PyTorch as fallback for GPU memory management
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for cuflash")

            O = torch.empty_like(Q)
            L = torch.empty((B, H, N), dtype=torch.float32, device=Q.device)

            result = self.lib.cuflash_attention_forward_f32(
                ctypes.c_void_p(Q.data_ptr()),
                ctypes.c_void_p(K.data_ptr()),
                ctypes.c_void_p(V.data_ptr()),
                ctypes.c_void_p(O.data_ptr()),
                ctypes.c_void_p(L.data_ptr()),
                B, H, N, D,
                ctypes.c_float(scale),
                causal,
                None  # stream: None = default CUDA stream
            )
        
        self.check_error(result)
        return O, L


def demo_numpy_cuda():
    """Demo using NumPy with CUDA via CuPy."""
    print("=" * 60)
    print("Demo: cuflash with CuPy")
    print("=" * 60)
    
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not available. Skipping this demo.")
        print("Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
        return
    
    attn = CuFlashAttn()
    
    # Small test case
    B, H, N, D = 2, 8, 512, 64
    print(f"\nTest configuration: Batch={B}, Heads={H}, Seq={N}, Dim={D}")
    
    # Create random GPU arrays
    Q = cp.random.randn(B, H, N, D).astype(cp.float32)
    K = cp.random.randn(B, H, N, D).astype(cp.float32)
    V = cp.random.randn(B, H, N, D).astype(cp.float32)
    
    print("Running forward pass...")
    O, L = attn.forward(Q, K, V, causal=True)
    
    print(f"Output shape: {O.shape}")
    print(f"Output mean: {float(cp.mean(O)):.6f}")
    print(f"Output std: {float(cp.std(O)):.6f}")
    print("✓ Forward pass completed successfully!\n")


def demo_pytorch_comparison():
    """Demo comparing cuflash with PyTorch native attention."""
    print("=" * 60)
    print("Demo: Comparison with PyTorch Native Attention")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch not available. Skipping this demo.")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping this demo.")
        return
    
    attn = CuFlashAttn()
    
    # Test configuration
    B, H, N, D = 2, 8, 1024, 64
    print(f"\nTest configuration: Batch={B}, Heads={H}, Seq={N}, Dim={D}")
    
    # Create random tensors
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    
    # cuflash forward
    print("Running cuflash forward pass...")
    O_flash, L = attn.forward(Q, K, V, causal=True)
    
    # PyTorch reference (causal mask)
    print("Running PyTorch reference...")
    scale = 1.0 / np.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    O_torch = torch.matmul(attn_weights, V)
    
    # Compare
    max_diff = torch.max(torch.abs(O_flash - O_torch)).item()
    mean_diff = torch.mean(torch.abs(O_flash - O_torch)).item()
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    if max_diff < 1e-3:
        print("✓ Results match within tolerance!\n")
    else:
        print("⚠ Results differ significantly\n")


def benchmark_performance():
    """Simple performance benchmark."""
    print("=" * 60)
    print("Benchmark: cuflash Performance")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("PyTorch not available. Skipping benchmark.")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    attn = CuFlashAttn()
    torch.manual_seed(42)
    
    configs = [
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
    ]
    
    print("\nConfig (B, H, N, D) | Time (ms)")
    print("-" * 40)
    
    for B, H, N, D in configs:
        Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
        K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
        V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
        
        # Warmup
        for _ in range(5):
            _, _ = attn.forward(Q, K, V, causal=True)
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(10):
            _, _ = attn.forward(Q, K, V, causal=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10 * 1000
        
        print(f"({B}, {H}, {N}, {D})      | {elapsed:7.3f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("cuflash Python Binding Examples")
    print("=" * 60 + "\n")
    
    try:
        # Demo 1: CuPy integration
        demo_numpy_cuda()
        
        # Demo 2: PyTorch comparison
        demo_pytorch_comparison()
        
        # Demo 3: Benchmark
        benchmark_performance()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Built the library: cmake --preset release && cmake --build --preset release")
        print("2. Installed dependencies: pip install numpy cupy (or torch)")
        sys.exit(1)
    
    print("All demos completed successfully!")
