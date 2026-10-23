# EqDeepRx：Fig. 6(a) 解码前 BER 复现

本仓库提供 EqDeepRx 实现、70,000 步训练断点、逐步训练记录，以及 CDL-C 场景下 Fig. 6(a) 的解码前 BER 仿真结果。图中包含 EqDeepRx、传统接收机和已知信道参考曲线；具体数值与误码/比特计数见 [结果数据](artifacts/strict_fig6a/figure6a_2layers_corrected/figure6a_metrics.json)。

![CDL-C 场景下的解码前 BER](artifacts/strict_fig6a/figure6a_2layers_corrected/figure6a_uncoded_ber.png)

## 仓库文件

| 路径 | 内容 |
|---|---|
| `eqdeeprx/` | 信号生成、信道、接收机、网络、损失函数、训练与评估代码 |
| `evaluate_fig6a_corrected.py` | 使用随仓库提供的断点进行 Fig. 6(a) 评估，并核对断点及训练源码哈希 |
| `artifacts/strict_fig6a/eqdeeprx.pt` | 70,000 步联合模型与优化器断点 |
| `artifacts/strict_fig6a/training.jsonl` | 逐步训练记录 |
| `artifacts/strict_fig6a/training_source_manifest.json` | 断点和训练时 Python 源码的 SHA-256 清单 |
| `artifacts/strict_fig6a/preflight.json` | 此次训练使用的运行环境与批量配置 |
| `artifacts/strict_fig6a/figure6a_2layers_corrected/` | 正式评估的 PNG、指标 JSON 与可续算进度 JSON |

训练断点的 SHA-256 为 `29dbc07846d748d630fc0ddce41156b06a3d25f46e9a41484015748d8c5a228b`，其 `next_step` 为 70,000。训练后仅对传统基线的评估处理顺序作了修正：先对导频信道估计线性插值，再在频域使用固定 5 抽头平滑。`evaluate_fig6a_corrected.py` 核对其他训练模块与原始源码清单一致，并使用修正后的评估代码；从头训练的新断点使用当前源码，可直接通过 `python -m eqdeeprx evaluate` 评估。

## 环境

记录本次运行的环境为 Windows、Python 3.12.14、NVIDIA GeForce RTX 5060 Laptop GPU、PyTorch 2.11.0+cu128、Sionna 2.1.0、NumPy 2.5.3、SciPy 1.18.1 和 Matplotlib 3.11.1。依赖版本固定在 [requirements.txt](eqdeeprx/requirements.txt)。以下命令在 PowerShell 中执行，并将训练与评估生成的数据放在 D 盘。

```powershell
git clone https://github.com/Tonyerwite/EqDeepRX.git
Set-Location .\EqDeepRX
py -3.12 -m venv D:\EqDeepRxRuntime\sionna-venv
$py = 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe'
& $py -m pip install -r .\eqdeeprx\requirements.txt
```

## 使用仓库断点复算 Fig. 6(a)

将断点和源码清单复制到 D 盘独立运行目录。先执行只读校验，再进行每种 DMRS 配置 32,000 个独立时隙的完整评估；结果写入 `$runDir\figure6a_2layers_corrected`。评估中断后，在相同运行目录的命令末尾添加 `--resume`。

```powershell
$runDir = 'D:\EqDeepRxRuns\from_repository'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Copy-Item .\artifacts\strict_fig6a\eqdeeprx.pt $runDir
Copy-Item .\artifacts\strict_fig6a\training_source_manifest.json $runDir
& $py .\evaluate_fig6a_corrected.py --run-dir $runDir --verify-only
& $py .\evaluate_fig6a_corrected.py --run-dir $runDir --samples 32000 --layers 2
```

正式评估使用种子 1,002,026。图示 SINR 中心点为 −5 至 13 dB、间隔 2 dB，每个点汇总其中心左右各 1 dB 的时隙。单导频与双导频各生成 32,000 个时隙，共 64,000 个；其中 28,212 个时隙位于所绘制的十个 SINR 区间。每个曲线点的时隙数、误码数、比特数与 BER 均保存在指标 JSON 中。

## 从头训练和评估

在新的 D 盘运行目录先执行预检，再以有效批量 112 训练 70,000 步。预检会记录当前 GPU、依赖版本、显存设置、源码哈希和可用计算批量；训练会据此核对运行条件。检查点每 50 个完成步骤保存一次。

```powershell
$trainDir = 'D:\EqDeepRxRuns\from_scratch'
& $py -m eqdeeprx preflight --run-dir $trainDir --microbatch-size 56 --generation-batch-size 2 --repeats 3
& $py -m eqdeeprx train --run-dir $trainDir --steps 70000 --save-every 50 --confirm
```

中断后在同一目录续训；训练完成后使用当前源码的普通评估入口。输出图和指标位于 `$trainDir\figure6a_2layers`。

```powershell
& $py -m eqdeeprx train --run-dir $trainDir --steps 70000 --save-every 50 --confirm --resume
& $py -m eqdeeprx evaluate --run-dir $trainDir --samples 32000 --layers 2
```

| 训练配置 | 数值 |
|---|---|
| 信道与波形 | UMa 在线生成；OFDM，30 kHz 子载波间隔，192 个子载波，14 个 OFDM 符号 |
| 天线与层数 | 16 个接收天线；每个 UE 1 个发送天线；训练覆盖 2、3、4 层，各以 1/3 概率采样 |
| 调制与导频 | 64-QAM；1 或 2 个 DMRS 符号，各以 1/2 概率采样；每 4 个子载波设置正交导频 |
| 信道条件 | SNR 在 0–45 dB 均匀采样；UE 速度在 0–35 m/s 均匀采样；0/1 个干扰 UE 各以 1/2 概率采样；INR 的 dB 值为均值 10、标准差 5 的正态分布 |
| 优化 | LAMB；有效批量 112；初始学习率 0.0044，70,000 步线性衰减至 0；种子 2026；无信道编码训练 |
| 显存执行 | 56 时隙梯度累积块，信道生成批量 2；整批 112 时隙共同完成一次优化器更新；bfloat16 卷积、关键接收机和损失计算使用 float32 |
| 模型 | DenoiseNN 宽度 `[64,64,64,2]`；DetectorNN 4 个 section、每层 64 通道；DemapperNN 宽度 `[32,32,32,8]`；time-mixer 通道数 2 |

表中的层数/导频/干扰抽样概率与 time-mixer 通道数是此实现的固定设置；论文给出了相应训练情形或结构，但未列出这些设置的数值。其余系统细节见 [`eqdeeprx/config.py`](eqdeeprx/config.py)。

## Fig. 6(a) 仿真场景

| 项目 | 本仓库结果 |
|---|---|
| 信道 | 3GPP CDL-C；时延扩展在 10–1100 ns 均匀采样 |
| UE 速度与干扰 | 10–15 m/s；1 个独立信道、随机符号偏移的干扰 UE |
| 空间层数 | 2 层；训练模型同时覆盖 2、3、4 层。论文 Fig. 6(a) 未单独标注层数，本图明确采用 2 层 |
| 波形与接收 | 30 kHz OFDM、192 子载波、14 符号、16 接收天线、64-QAM；传统基线使用导频估计、频域平滑及 LMMSE 接收；绿色曲线使用已知信道 |
| 导频与编码 | 1/2 个 DMRS 符号；发送端分别使用 MCS Table 2 的索引 11/12 进行 LDPC 编码；接收端在 LDPC 译码前计算 BER |
| 统计 | 每种导频 32,000 个独立时隙；按每时隙实际 SINR 分箱；误码数除以对应编码比特数 |

论文中 Fig. 6(a) 同时画有单独 DenoiseNN 曲线；本仓库的结果图只绘制 EqDeepRx、传统基线和已知信道参考。`figure6a_metrics.json` 是图的数值来源，包含各 SINR 区间的样本数、误码数及比特数。

## 参考文献

M. Honkala, D. Korpi, E. Raninen, and J. M. J. Huttunen, “EqDeepRx: Learning a Scalable and Interference Mitigating MIMO Receiver,” arXiv:2602.11834, 2026. [论文](https://arxiv.org/abs/2602.11834)。
