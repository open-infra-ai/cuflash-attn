# cuflash Copilot Instructions

- 用中文回复；code comments 和 API docs 保持英文。
- 这是一个精简维护中的 CUDA FlashAttention 参考库；优先保持实现清晰、稳定、可审计。
- 保持以下不变量：张量布局 `[batch, heads, seq_len, head_dim]`，`head_dim` 仅支持 `32`、`64`、`128`。
- 始终使用 CMake presets：`cmake --preset release`、`cmake --build --preset release`、`ctest --preset release --output-on-failure`。
- 所有 CUDA 资源调用都要检查错误；公开 API 返回 `FlashAttentionError`，不要抛异常。
- 优先做小而完整的改动；删掉冗余流程、重复文档和无效约束，不引入新的 AI 控制框架。
