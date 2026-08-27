# Domain Context

cuflash 是一个 CUDA C++ FlashAttention 实现库。本文档定义核心领域术语，供架构讨论和代码审查使用。

## Core Concepts

### FlashAttention
分块式注意力算法，实现 O(N) 内存复杂度。核心思想是将 Q、K、V 分成小块（tiles）在 shared memory 中计算，避免实例化完整的 N×N attention matrix。

### Tile
FlashAttention 中的计算单位。一块 Q、K 或 V 的行数据，大小由 `BLOCK_M`、`BLOCK_N` 等模板参数定义。Tile 在 shared memory 中处理，支持向量化和边界处理。

### Online Softmax
数值稳定的流式 softmax 算法。维护两个状态：
- `m`: 当前最大值
- `l`: exp(x - m) 的累加和

支持增量式更新，无需存储完整 softmax matrix。

使用 `impl::OnlineSoftmaxState` 结构体统一实现，确保前向和反向传播使用相同的数值算法。

### Matmul Operations
Tile 级别的矩阵乘法原语：
- `matmul_ABt`: C = A @ Bᵀ（attention score 计算）
- `matmul_AB`: C = A @ B
- `matmul_AB_acc`: C += A @ B（累加）
- `matmul_AtB`: C = Aᵀ @ B

### Workspace
反向传播所需的中间 GPU 内存缓冲区，存储 D 数组（denominator）。

### Kernel
CUDA 设备函数，在 GPU 上并行执行。FlashAttention 的 kernel 是参数化的，支持不同 `HEAD_DIM`（32, 64, 128）。

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│  cuflash::flash_attention_forward/backward      │  ← 高级 API
├─────────────────────────────────────────────────┤
│  cuflash::kernels::*                            │  ← Kernel 原语（公开，用于测试）
│    - online_softmax_init/update/finalize        │
│    - matmul_ABt, matmul_AB, matmul_AB_acc, AtB  │
│    - load_tile, store_tile                      │
├─────────────────────────────────────────────────┤
│  src/kernels/impl/*                             │  ← 实现细节（内部）
│    - OnlineSoftmaxState (device struct)         │
│    - ForwardTilingConfig, BackwardTilingConfig  │
│    - __device__ matmul functions                │
│    - __device__ tile I/O functions              │
└─────────────────────────────────────────────────┘
```

## Tiling Configuration

Tiling 参数集中定义在 `src/kernels/impl/tile_io.cuh` 中：

- **ForwardTilingConfig**: 前向传播使用较大的 block（64x64，head_dim=128 时用 32x32）
- **BackwardTilingConfig**: 反向传播使用较小的 block（64x64，head_dim=128 时用 16x32）

差异原因：反向传播需要在 shared memory 中存储更多梯度张量（dQ, dK, dV），因此需要更保守的 tiling 策略。

## Key Invariants

1. **Tensor Layout**: `[batch_size, num_heads, seq_len, head_dim]` — 不可变
2. **Supported head_dim**: 32, 64, 128 — 由 `impl::is_supported_head_dim()` 集中检查
3. **Data Types**: FP32 (float) and FP16 (half) — 内部计算始终用 float
4. **Stream Safety**: 所有 CUDA 操作使用显式 stream 参数

## Design Principles

1. **Depth over Shallow**: Kernel utilities 有公开接口，测试不穿透实现细节
2. **Primitive Decomposition**: 复杂操作分解为可组合的原语
3. **Template for Performance**: M, N, K 作为编译期模板参数，确保 kernel 优化
4. **Centralized Configuration**: Tiling 配置和支持的 head_dim 值集中定义，避免分散
