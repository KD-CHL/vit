# Oxford-IIIT Pet 视觉模型微调

在 37 类猫狗品种分类上比较随机初始化的 ResNet-18、ViT-B/16 线性探测、ViT-B/16 全量微调，以及 ViT-B/16 + LoRA。评估使用 Oxford-IIIT Pet 官方测试集。

## 复现实验

需要 Python 3.10（实测版本）、支持 CUDA 12.8 的 NVIDIA 驱动，以及至少 15 GB 可用磁盘。配置按 RTX 5060 Ti 8 GB 的条件设置。训练权重、数据档案和特征缓存保存在本地。

**Windows PowerShell**

```powershell
git clone https://github.com/KD-CHL/vit.git
Set-Location vit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:VIT_RESULT_ROOT = 'reproduction'
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.run_suite
```

**Linux 或 Git Bash**

```bash
git clone https://github.com/KD-CHL/vit.git
cd vit
python -m venv .venv
source .venv/bin/activate             # Git Bash 用 source .venv/Scripts/activate
python -m pip install -r requirements.txt
bash scripts/01_prepare_data.sh
bash scripts/02_run_all.sh
```

`src.prepare` 下载并校验数据档案和 timm 权重，按种子 42 分层划分训练和验证集，随后核对仓库中的样本清单，并生成数据统计和训练样例图。

`src.run_suite` 按种子 42、123、2026 运行四组主实验，再以种子 42 跑 CLS/均值池化、数据增强和 LoRA 秩值消融。已完成的运行会跳过。中断后重跑原命令会从最近完整保存的一轮续训。训练跑完后，脚本用每组最佳 checkpoint 评估测试集并生成报告。可以用 `--train-only` 运行到训练结束，或用 `--workers 4` 减少并行加载进程。

以上命令将本次运行写入 `reproduction/`。其中 `outputs/` 存放实验数据、`runs/` 存放本地权重和 TensorBoard 日志，`report.md` 为本次运行的汇总。

| 路径 | 内容 |
| --- | --- |
| `configs/` | 四组模型的训练配置 |
| `splits/` | 训练、验证、测试 CSV 和品种字典 |
| `outputs/data/` | 图像解码排查、尺寸分布及各类样例 |
| `outputs/training/` | 各种子的逐轮日志、峰值显存和吞吐量 |
| `outputs/evaluation/` | 每个 checkpoint 的逐图预测、逐类指标及混淆矩阵 |
| `outputs/analysis/` | Grad-CAM 图、20 张错分样例、LoRA 合并推理记录 |
| `report.md` | 实验设置、对比结果和错误分析 |

## 推理

```bash
python -m src.predict --checkpoint reproduction/runs/v4_s42/best.pt --input /path/to/image.jpg
python -m src.predict --checkpoint reproduction/runs/v4_s42/best.pt --input /path/to/images --output reproduction/predictions.json
```

输出为每张图片的 top-3 品种与 softmax 置信度。V4 默认调用 PEFT 的 `merge_and_unload` 后推理，`--no-merge` 可保留适配器以核对输出。TensorBoard 用 `tensorboard --logdir reproduction/runs` 查看。
