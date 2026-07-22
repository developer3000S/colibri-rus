# DeepSeek V4（colibri CPU）

[English](deepseek-v4.md) · 简体中文

colibri 上的 DeepSeek V4 Flash + DSpark 本地 CPU 推理：涵盖权重加载、
专家缓存、稀疏注意力、推测解码和自动 RAM 分层，端到端生成已通过冒烟测试。

## 已完成

- **目标模型引擎**：配置／层／注意力／块／专家存储／数学运算／运行时／
  prompt；FP8 稠密权重与原生 FP4 专家内核；在硬件支持时启用 AVX-512
  批量验证和 rows16 热门专家布局。
- **DSpark 推测解码**：草稿运行器、heads、验证窗口、prefix/commit；
  在冒烟测试 prompt 上接受率达到 100%。
- **自动 RAM 策略**：根据操作系统可用内存或 `--ram`，规划常驻张量、
  DSpark stages、专家缓存和输出 head。
- **SSD 专家流式加载**：每个专家使用两次合并读取（scales + weights）；
  无需重新排列成 160 GB 容器。
- **源码布局**：采用与 GLM 类似的聚合源文件 `deepseek_v4.c`／
  `deepseek_v4_dspark.c`；公共 API 位于 `deepseek_v4.h`
  （`ColiV4Engine`／`ColiV4Session`／配置／prompt）；实现细节位于
  `deepseek_v4_internal.h`，不承诺接口稳定性。
  `make deepseek-v4` 会构建 `c/deepseek_v4.exe`。
- **CLI**：`c/v4` 是仅使用标准库的 Python 启动器，封装 `run`／`chat`；
  推理本身由 engine + session 完成（`coli_v4_engine_open` →
  `coli_v4_session_create`／`generate` → 销毁所有 session，再销毁 engine）。
  `deepseek_v4.h` 中的 engine/session 接口是**实验性**公共 API，未来可能变化。
  Engine 会复制模型目录字符串；调用 `coli_v4_engine_destroy` 前必须销毁所有
  session。Index 与 ExpertStore 访问器仍是内部接口
  （`deepseek_v4_internal.h`）。

## 模型概况

典型 `DeepSeek-V4-Flash-DSpark` 结构（来自 checkpoint 元数据）：

| 项目 | 数值 |
|------|-----:|
| Transformer 层数 | 43 |
| Hidden size | 4096 |
| 每层路由专家数 | 256（top-k 6） |
| 每条专家记录大小 | 约 12.75 MiB（FP4 E2M1 + UE8M0 scales） |
| 滑动窗口 | 128 |
| 稠密层常驻大小 | 约 6.27 GiB |
| 路由专家 payload | 约 137 GiB（流式加载，不会全部常驻） |

专家使用 packed FP4（I8 safetensors 中的 `float4_e2m1fn_x2`），并非 INT8。
稠密权重使用带 UE8M0 block scales 的 E4M3。参考路径以 FP8×FP4 计算激活，
并使用 FP32 累加。

## 自动 RAM 规划

预算分配顺序：系统保留 → 运行时工作集 → 最小专家 slots
（top-k × 稀疏层数）→ 可选的常驻 BF16 `lm_head`（约 1.06 GiB）→
用剩余预算增加每层专家缓存 slots。

- 未指定 `--ram` 时，规划器使用**操作系统可用内存**而非总内存，并保留
  自适应系统余量。
- 指定 `--ram GiB` 时，该数值是**规划预算**，不是操作系统强制执行的硬上限；
  它控制预计常驻权重、缓存和运行时工作区。
- 如果最小专家工作集无法容纳，启动会明确报告差额并失败，不会静默依赖
  swap 超额分配。
- 输出 head 是否常驻是可选的：低内存模式会流式读取 BF16 head；预算充足时
  则常驻以提高速度。
- 热门专家可固定并原地重排为 16-row AVX-512 布局；冷门专家仍使用官方的
  row-major FP4。重排不会增加容量占用。

显式指定 32 GiB 规划预算时的常驻示例：

```text
dense ≈ 6.27 GiB
dspark ≈ 10.45 GiB
target expert cache ≈ 9.1 GiB（RAM 允许时增加 slots）
head = resident BF16
```

在**宿主机**有足够空闲内存容纳最小工作集时，规划器也能根据 8 GiB 预算
生成进程方案；由于专家缓存更紧，解码会更慢。这不等同于可以在实体 8 GiB
电脑上交付运行，详见下面的基准测试说明。

## 基准测试

以下数据使用的硬件：

| 项目 | 数值 |
|------|------|
| CPU | AMD Ryzen AI MAX+ 395（16 核／32 线程，Radeon 8060S） |
| 系统内存 | 128 GiB |
| 操作系统 | Windows 11 |
| 构建 | `make deepseek-v4`（MSYS2 UCRT64，`-march=x86-64-v3`） |

模型：`DeepSeek-V4-Flash-DSpark`

Prompt：`--stop-sentence "What is the capital of France?"`
（输出：*The capital of France is Paris.*）

这里的 `--ram` 是高内存机器上的**规划预算**，不是 RSS 硬上限，也不代表
实体 8 GiB／32 GiB 电脑。**8 GiB 一行只是规划设置**（`--ram 8`）：
宿主机仍有充足的操作系统内存／页面缓存供 SSD 专家 I/O 使用。实体约 8 GiB
系统可能无法启动，或在系统保留、运行时临时空间和磁盘缓存争夺同一预算时
运行得慢得多；请勿把该行当作 8 GiB 硬件保证。

| `--ram` | TTFT | prefill | decode | DSpark 接受率 |
|---------|------|---------|--------|---------------|
| 32 GiB | 9.57s | 1.403 tok/s | 1.236 tok/s | 100% |
| 8 GiB（规划预算） | 10.02s | 1.338 tok/s | 0.712 tok/s | 100% |

```text
# --ram 32
[stats] RAM 31.98/32.00 GiB | TTFT 9.57s | prefill 1.403 tok/s | decode 1.236 tok/s | DSpark acceptance 100.0%

# --ram 8（约 127 GiB 宿主机上的规划预算，并非实体 8 GiB 机器）
[stats] RAM 7.55/8.00 GiB | TTFT 10.02s | prefill 1.338 tok/s | decode 0.712 tok/s | DSpark acceptance 100.0%
```

## 构建

V4 引擎及其聚合源文件单元测试支持以下平台：

- x86-64 Linux（gcc；硬件支持时使用 AVX-512 路径）
- Windows／MSYS2 UCRT64（同上）

macOS、PowerPC 和其他主机仍通过 `make check` 验证 colibri 主引擎，
但**不会**构建或链接 DeepSeek V4。在不支持的平台上，
`make deepseek-v4` 会给出明确错误并退出。

```bash
# MSYS2 UCRT64
export PATH=/ucrt64/bin:/usr/bin
cd /d/ai/colibri/c
make deepseek-v4
```

也可以在仓库根目录运行：`make deepseek-v4` → `c/deepseek_v4.exe`。

聚合源文件按照原先的顶层单元，使用 `-DCOLI_V4_UNIT_*` 分别编译，确保多层
`#include` 与宏封装保持正确。共享内核（`native_quant*`、
`safetensors_index`、`tensor_io`）仍是独立的编译单元。

## 已提交的 tiny 独立 oracle

x86-64 Linux 和 Windows／MSYS2 上的 `make check` 会构建 V4 引擎并运行
`make deepseek-v4-tiny-check`。仓库中的 `c/deepseek_v4_tiny` fixture 是有效的
缩小版 checkpoint，包含一个 DSpark stage，大小约 1.2 MiB。测试时无需网络、
PyTorch、Transformers 或下载模型。不支持的平台会跳过 V4 执行，同时保留
普通 GLM 构建和平台门控检查。

`deepseek_v4_tiny/ref.json` 中的参考结果由官方 Transformers
`DeepseekV4ForCausalLM` 生成，并经过 C checkpoint 使用的稠密 FP8、路由专家
FP4 和 BF16 权重 round trip。这个零依赖测试比较整数 token ID，而不是解码后的
文本，覆盖：

- 目标模型 teacher forcing 与仅目标模型的 greedy decoding；
- 禁用 drafting 时的目标模型 session 生成；
- DSpark proposal 加目标模型验证，并保证与目标路径精确一致；
- 滑动、压缩稀疏和高度压缩注意力；
- 一个 72-token prompt，跨越内部 64-token prefill 边界；
- 重复执行 engine/session 的打开、生成和销毁生命周期。

Fixture 再生成刻意与 CI 分离。仓库中的 fixture 使用 Python 3.12、
PyTorch 2.13.0、Transformers 5.14.1 和 safetensors 0.8.0 创建
（safetensors 作为 Transformers 依赖安装）：

```bash
python -m pip install torch==2.13.0 transformers==5.14.1 safetensors==0.8.0
python c/tools/make_deepseek_v4_tiny.py --force
make -C c deepseek-v4-tiny-check
```

生成器没有 C 引擎 fallback：如果官方 DeepSeek V4 支持不可用，它会直接失败；
同时会打印每个张量的名称与 shape，并在 `ref.json` 中记录 schema/generator、
PyTorch 和 Transformers 版本。

这些检查证明缩小量化模型的 top-1 token 一致性，并执行真实的
target/DSpark/session 路径；但不能证明 logit 级一致性、生产 checkpoint 性能、
所有专家／缓存常驻策略或完整模型质量。

要在禁用 drafting 的情况下执行普通运行时：

```text
./deepseek_v4 deepseek_v4_tiny '<t005><t007><t009>' \
  --raw-prompt --draft-model deepseek_v4_tiny/dspark --no-dspark
```

参考来源的区别：

- 已提交的 tiny fixture：独立的 `source=transformers` oracle；
- 下方完整 checkpoint 冒烟测试：`source=coli-self` 一致性 oracle。

## 完整 checkpoint oracle 验证（不属于轻量 CI）

Upstream 要求有针对性的 token 级检查。V4 单元测试覆盖配置／存储／数学运算；
完整模型正确性使用单独的重型路径验证：

```bash
cd c
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash-DSpark MEMORY_GB=32
# 或者：
python tools/make_deepseek_v4_oracle.py \
  --model /path/to/DeepSeek-V4-Flash-DSpark \
  --binary ./deepseek_v4.exe \
  --output tests/deepseek_v4_oracle.json \
  --validate --teacher-forcing 32 --greedy 20 --check-dspark
```

```text
./deepseek_v4 MODEL --oracle tests/deepseek_v4_oracle.json \
  --teacher-forcing 32 --greedy 20 --memory-gb 32
```

### 比较约定

| 检查 | 判定标准 |
|------|----------|
| Teacher-forcing | top-1 token 精确一致：N/N 个位置与 `tf_pred` 比较 |
| Greedy | top-1 token 精确一致：N/N 个续写 token 与 `full_ids` 比较 |
| DSpark 开／关 | greedy token 序列相同（`--check-dspark`） |
| Logits／top-k | 仅用于 `source=transformers` fixture |

Fixture 的 `source` 字段：

- `coli-self`——由 C 引擎使用 `--no-dspark` 记录（Hugging Face DeepSeek V4
  不可用时的默认方式）。它证明结果可复现，并证明推测解码与目标模型 greedy
  路径一致；**不代表**与 HF bit-exact。
- `transformers`——成功加载 `DeepseekV4ForCausalLM` 时使用的官方实现
  （`--prefer-transformers`）。

完整 checkpoint oracle JSON 是本地生成文件，不是 `make check` 的必需项；
已提交的 tiny 独立 oracle 则是必需项。

## 运行

```powershell
cd D:\ai\colibri
python ./c/v4 run --model D:/ai/DeepSeek-V4-Flash-DSpark --ram 32 `
  --stop-sentence "What is the capital of France?"
```

也可以直接调用引擎：
`.\c\deepseek_v4.exe <model> <prompt> --max-tokens N --memory-gb G …`。

### 选项

- `--model PATH`（必需）：DeepSeek V4 checkpoint 目录
- `--ram GiB`：规划预算；省略时根据操作系统可用内存自适应规划
- `--ngen N`：最大生成 token 数，默认 128
- `--stop-sentence`（`run`）：在第一个句子终止符处停止
- `--system`／`--thinking`：仅适用于 `chat`

Chat 每轮都会重新 prefill 历史记录；尚未实现跨轮 KV 复用。
默认的非 thinking 编码为：

```text
<｜begin▁of▁sentence｜>[system]<｜User｜>[user]<｜Assistant｜></think>
```

## 待办事项

- [ ] **session parallel-prefix**：将
      `COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY` 接入
      `coli_v4_session_generate`，达到旧版 CLI 的验证吞吐
- [ ] **16 GiB 性能**：缩小与 32 GiB 的解码差距（缓存、固定、I/O 重叠）
- [ ] **模型量化**：采用更激进的权重／激活路径，降低占用和带宽需求
- [ ] **服务器**：面向多客户端的 HTTP／OpenAI 兼容 API
- [ ] **CUDA**：在 CPU 路径之外增加可选 GPU 后端
- [ ] **长输出动态 DSpark 调节**：根据近期 prefix survival／confidence，
      在 K=2–4 之间动态选择验证窗口，不再始终使用固定 K
- [ ] Chat 跨轮 KV cache 复用，避免每轮重新 prefill 全部历史
