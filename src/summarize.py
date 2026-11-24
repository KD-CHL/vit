import csv
import statistics as stats
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import OUTPUTS, ROOT, SEEDS, read_json, save_json

LABELS = {
    "v1": "V1 ResNet-18 从零", "v2": "V2 ViT-B/16 线性探测",
    "v3": "V3 ViT-B/16 全量微调", "v4": "V4 ViT-B/16 + LoRA (r=8)",
}
COLORS = ("#185b89", "#bc6238", "#438060")
GIB = 2 ** 30


def draw_curves(experiment, histories, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    for seed, history, color in zip(SEEDS, histories, COLORS):
        epochs = [row["epoch"] for row in history]
        for key, ax in zip(("loss", "top1"), axes):
            for split, style in (("train", "-"), ("val", "--")):
                data = [row[split][key] for row in history]
                ax.plot(epochs, data, style, color=color, label=f"{seed} {split}", lw=1.35)
    axes[0].set(ylabel="Cross-entropy loss", xlabel="Epoch")
    axes[1].set(ylabel="Top-1 accuracy", xlabel="Epoch")
    axes[1].legend(ncol=2, fontsize=8, loc="best")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / f"{experiment}_curves.png", dpi=150)
    plt.close(fig)


def load_runs():
    result = {}
    curve_dir = OUTPUTS / "curves"
    curve_dir.mkdir(exist_ok=True)
    for experiment in LABELS:
        runs, histories, metrics = [], [], []
        for seed in SEEDS:
            name = f"{experiment}_s{seed}"
            runs.append(read_json(OUTPUTS / "training" / name / "summary.json"))
            histories.append(read_json(OUTPUTS / "training" / name / "history.json"))
            metrics.append(read_json(OUTPUTS / "evaluation" / name / "metrics.json"))
        draw_curves(experiment, histories, curve_dir)
        result[experiment] = (runs, histories, metrics)
    return result


def summarize_metric(metrics, key):
    values = [item[key] for item in metrics]
    return {"mean": stats.mean(values), "sd": stats.stdev(values)}


def summarize_main(all_runs):
    summary = {}
    for key, (runs, histories, metrics) in all_runs.items():
        summary[key] = {
            "parameters_trainable": runs[0]["parameters_trainable"],
            "parameters_total": runs[0]["parameters_total"],
            "peak_gib": max(run["peak_bytes"] for run in runs) / GIB,
            "median_epoch_seconds": stats.median(run["median_train_seconds"] for run in runs),
            "median_images_per_second": stats.median(run["median_train_ips"] for run in runs),
            "median_wait_fraction": stats.median(
                row["train"]["loader_wait_seconds"] / row["train"]["seconds"]
                for history in histories for row in history[1:]
            ),
            "top1": summarize_metric(metrics, "top1"),
            "top5": summarize_metric(metrics, "top5"),
            "macro_f1": summarize_metric(metrics, "macro_f1"),
            "species_error_rate": summarize_metric(metrics, "species_error_rate"),
        }
    return summary


def ablation_tables():
    experiment_rows = {}
    ablations = {
        "线性探测的特征池化": [("CLS", "v2"), ("Patch token 均值", "v2_mean")],
        "全量微调的数据增强": [("无增强", "v3_none"), ("裁剪与翻转", "v3_basic"),
                                  ("裁剪、翻转与 RandAugment", "v3")],
        "LoRA 秩值": [("r=4", "v4_r4"), ("r=8", "v4"), ("r=16", "v4_r16")],
    }
    for title, comparisons in ablations.items():
        experiment_rows[title] = [
            (label, read_json(OUTPUTS / "training" / f"{run}_s{SEEDS[0]}" / "summary.json"))
            for label, run in comparisons
        ]
    return experiment_rows


def add_data(report, data):
    counts = data["counts"]
    modes = "、".join(f"{mode} {count:,} 张" for mode, count in data["color_modes"].items())
    shape = data["geometry"]
    report += [
        "## 1 数据与划分",
        "",
        f"Oxford-IIIT Pet 含 37 个品种，其中猫 12 类、狗 25 类。用固定种子 "
        f"{data['split_seed']} 从官方 trainval 分层抽取 10% 作为验证集，训练、验证和测试"
        f"分别为 {counts['train']:,}、{counts['val']:,} 和 {counts['test']:,} 张。"
        "品种标签及猫狗大类映射来自官方标注。"
        "[Oxford-IIIT Pet 数据集](https://www.robots.ox.ac.uk/~vgg/data/pets/)",
        "",
        f"原图中位尺寸为 {shape['width'][2]:.0f} × {shape['height'][2]:.0f} 像素"
        "（宽、高分别取中位数），宽高比的第 5 至 95 百分位为 "
        f"{shape['aspect_ratio'][1]:.2f} 至 {shape['aspect_ratio'][3]:.2f}。"
        f"解码排查共排除 {len(data['invalid_images'])} 张图。"
        f"有效图像原始模式为 {modes}，读取时统一转为 RGB。"
        "每类样本数、分位数和排除记录见下表及数据统计。"
        "[数据统计 JSON](outputs/data/summary.json)",
        "",
        "![图像尺寸及宽高比分布](outputs/data/image_geometry.png)",
        "",
        "![训练集 37 类各 4 张样例](outputs/data/train_examples.jpg)",
        "",
        "观察训练样例时，优先检查短毛灰色猫品种（British Shorthair、Russian Blue）、"
        "重点色或长毛猫品种（Birman、Ragdoll），以及 Staffordshire Bull Terrier 与 "
        "American Pit Bull Terrier。实测错分配对见第 5 节。",
        "",
        "<details>",
        "<summary>37 类样本数</summary>",
        "",
        "| 品种 | 大类 | 训练 | 验证 | 测试 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for breed in data["breed_counts"]:
        species = "猫" if breed["species"] == "cat" else "狗"
        report.append(f"| {breed['breed']} | {species} | {breed['train']} | "
                      f"{breed['val']} | {breed['test']} |")
    report += ["", "</details>", ""]


def add_setup(report, all_runs):
    first = next(iter(all_runs.values()))[0][0]
    report += [
        "## 2 训练设置",
        "",
        f"在 {first['device']} 上运行。PyTorch {first['torch']}，CUDA {first['cuda']}，"
        f"AMP 使用 {first['amp'].split('.')[-1]}。主实验随机种子为 "
        f"{'、'.join(map(str, SEEDS))}。每次运行按验证集 top-1 选 checkpoint，同分取较早轮次。",
        "",
        "V2 至 V4 使用相同的 ViT-B/16 权重，权重在 ImageNet-21k 上预训练后又"
        "在 ImageNet-1k 上微调。输入尺寸、均值和标准差由 timm 的 "
        "`resolve_data_config` 读取。验证和测试均缩放短边至 248、中心裁剪至 224，"
        "图像训练使用 224 随机缩放裁剪、水平翻转和 RandAugment。"
        "V2 缓存特征时使用固定中心裁剪。"
        "[timm 权重说明](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)",
        "",
        "| 实验 | 轮数 | 单步批量 × 累积步数 | 优化器 | 学习率 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for key, (runs, _, _) in all_runs.items():
        config = runs[0]["config"]
        lr = f"{config['lr']:.1g}"
        if "head_lr" in config:
            lr = f"主干 {lr}，分类头 {config['head_lr']:.1g}"
        report.append(f"| {key.upper()} | {config['epochs']} | {config['batch_size']} × "
                      f"{config['accumulate']} | {config['optimizer']} | {lr} |")
    report += [
        "",
        "V1 使用动量 0.9、权重衰减 0.0005，预热 5 轮。V2 至 V4 使用权重衰减 "
        "0.05，预热 1 轮。各组预热后使用余弦衰减。V3 和 V4 使用 0.1 标签平滑、"
        "0.1 drop path 和梯度检查点。V4 在每个注意力块的 qkv 线性层注入 LoRA，"
        "将 alpha 设置为秩的两倍，LoRA dropout 为 0.1，分类头同时参与训练。",
        "",
    ]


def add_results(report, summary, all_runs):
    report += [
        "## 3 主实验结果",
        "",
        "测试集指标为三个种子的均值 ± 样本标准差。峰值显存取三个种子训练阶段的 "
        "`torch.cuda.max_memory_allocated` 最大值，单位 GiB。单轮耗时为每个种子"
        "训练轮耗时中位数的中位数，含数据读取和训练，验证、存盘另计。",
        "",
        "| 模型 | 可训练参数量 | 峰值显存（GiB） | 单 epoch（s） | 测试 top-1（%） | "
        "测试 top-5（%） | 测试 Macro-F1（%） | 猫狗误判率（%） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in summary.items():
        metrics = [f"{item[name]['mean'] * 100:.2f} ± {item[name]['sd'] * 100:.2f}"
                   for name in ("top1", "top5", "macro_f1", "species_error_rate")]
        report.append(f"| {LABELS[key]} | {item['parameters_trainable']:,} | {item['peak_gib']:.3f} | "
                      f"{item['median_epoch_seconds']:.2f} | {' | '.join(metrics)} |")
    feature = all_runs["v2"][0][0]["feature_extraction"]
    report += [
        "",
        f"V2 表中列的是分类头拟合开销。训练和验证特征一次性抽取耗时 "
        f"{feature['seconds']:.1f} s，前向抽取峰值为 {feature['peak_bytes'] / GIB:.3f} GiB，"
        "CLS 和均值特征共享同一次骨干前向。",
        "",
        "| 实验 | 中位吞吐量（图/s） | 取下一批的主机等待占比（%） |",
        "| --- | ---: | ---: |",
    ]
    for key, item in summary.items():
        report.append(f"| {key.upper()} | {item['median_images_per_second']:.1f} | "
                      f"{item['median_wait_fraction'] * 100:.1f} |")
    workers = all_runs["v1"][0][0]["workers"]
    report += [
        "",
        f"DataLoader 使用 {workers} 个 worker 和页锁定内存。等待占比排除每次运行"
        "的首轮启动开销，表示主机获取 batch 时阻塞的时间占单轮耗时的比例。"
        "GPU 异步计算可与这段时间重叠。",
        "",
        "### 训练与验证曲线",
        "",
    ]
    for key in LABELS:
        report += [f"![{key.upper()} 三个种子的损失与准确率](outputs/curves/{key}_curves.png)", ""]
    v1_history = all_runs["v1"][1]
    gaps = [
        (history[-1]["train"]["top1"] - history[-1]["val"]["top1"]) * 100
        for history in v1_history
    ]
    v1_runs = all_runs["v1"][0]
    ending = stats.mean(history[-1]["val"]["top1"] for history in v1_history)
    best = stats.mean(run["best_val_top1"] for run in v1_runs)
    report += [
        f"V1 最后一轮训练 top-1 与验证 top-1 的平均差为 {stats.mean(gaps):+.2f} 个百分点。"
        f"验证集最佳 top-1 平均为 {best:.2%}，最后一轮验证 top-1 平均为 {ending:.2%}，"
        f"下降 {(best - ending) * 100:.2f} 个百分点。"
        "训练准确率在实时增强后的图片上计算。过拟合程度应结合验证曲线及最优轮次判断。",
        "",
    ]


def add_ablations(report, ablations):
    report += [
        "## 4 消融实验",
        "",
        "消融固定种子 42，使用验证集最佳 checkpoint 对比。增强消融沿用 V3 的其余训练设置，"
        "秩值消融沿用 V4 设置并固定 alpha/r = 2。主表预先固定 V2 使用 CLS、V3 使用 RandAugment、"
        "V4 使用 r=8。",
        "",
    ]
    for title, entries in ablations.items():
        report += [
            f"### {title}", "",
            "| 设置 | 最高验证 top-1 | 最佳轮次 | 可训练参数量 | 训练峰值显存（GiB） |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, run in entries:
            report.append(f"| {label} | {run['best_val_top1']:.2%} | {run['best_epoch']} | "
                          f"{run['parameters_trainable']:,} | {run['peak_bytes'] / GIB:.3f} |")
        report.append("")


def add_analysis(report, all_runs):
    metrics = all_runs["v3"][2][0]
    report += [
        "## 5 可视化与错误分析",
        "",
        "使用种子 42 的 V1 和 V3，对测试集固定抽取的相同 10 张图片做 Grad-CAM。"
        "V1 取最后一个残差块，V3 取最后一个注意力块的第一层归一化输出，"
        "ViT 热图由 14 × 14 的 patch 网格上采样。每张热图的目标是对应模型的 top-1。"
        "[图片清单](outputs/analysis/grad_cam_samples.json)",
        "",
        "![V1 与 V3 的 Grad-CAM](outputs/analysis/grad_cam_v1_v3.jpg)",
        "",
        "从 V3 种子 42 的错分结果中固定随机抽取 20 张。图中标注真实标签、模型预测和 "
        "top-1 置信度。逐类准确率、带方向的猫狗误判数及混淆矩阵均保存在"
        "每次运行的指标文件中。"
        "[错分样本清单](outputs/analysis/error_samples.csv) · "
        "[V3 种子 42 逐类指标](outputs/evaluation/v3_s42/metrics.json)",
        "",
        "![V3 的 20 张错分样本](outputs/analysis/v3_errors.jpg)",
        "",
    ]
    # Human observations are keyed by image path so they remain valid across reruns.
    observations = OUTPUTS / "analysis" / "error_observations.csv"
    if observations.exists():
        with observations.open(encoding="utf-8", newline="") as stream:
            reviewed = {row["path"]: row for row in csv.DictReader(stream)}
        with (observations.parent / "error_samples.csv").open(encoding="utf-8", newline="") as stream:
            ordered = list(csv.DictReader(stream))
        if {row["path"] for row in ordered} != set(reviewed):
            raise ValueError("Error observations do not match the selected sample paths.")
        counts = Counter(reviewed[row["path"]]["category"] for row in ordered)
        report.append("按主要可见问题归类，" + "、".join(
            f"{category} {count} 张" for category, count in counts.items()
        ) + "。")
        report += ["", "| 编号 | 标注 → 预测 | 主要类型 | 图像观察 |",
                   "| --- | --- | --- | --- |"]
        for row in ordered:
            item = reviewed[row["path"]]
            report.append(f"| {row['id']} | {row['label']} → {row['predicted']} | "
                          f"{item['category']} | {item['observation']} |")
        report.append("")
    cam_notes = OUTPUTS / "analysis" / "cam_observations.md"
    if cam_notes.exists():
        report += [cam_notes.read_text(encoding="utf-8").strip(), ""]
    report += [
        "### V3 高频混淆配对（种子 42）",
        "",
        "| 品种 A | 品种 B | A→B | B→A |",
        "| --- | --- | ---: | ---: |",
    ]
    for pair in metrics["top_confusions"][:5]:
        report.append(f"| {pair['class_a']} | {pair['class_b']} | {pair['a_as_b']} | {pair['b_as_a']} |")
    report += ["", "![V3 测试集混淆矩阵](outputs/evaluation/v3_s42/confusion.png)", ""]


def add_conclusions(report, summary):
    v1, v2, v3, v4 = (summary[key] for key in LABELS)
    gap = lambda a, b: (a["top1"]["mean"] - b["top1"]["mean"]) * 100
    demo = read_json(OUTPUTS / "analysis" / "lora_merged_prediction.json")
    report += [
        "## 6 实验结论与推理",
        "",
        f"1. V3 的测试 top-1 相对 V1 变化 **{gap(v3, v1):+.2f} 个百分点**。"
        "这个差值反映从随机初始化的 ResNet-18 切换到预训练 ViT-B/16 全量微调后的综合收益。"
        "两组同时改变了骨干、初始化和优化配置，单独估计预训练收益需要增加同架构的随机初始化对照。",
        f"2. 解冻 ViT 主干后，V3 的测试 top-1 相对 V2 变化 **{gap(v3, v2):+.2f} 个百分点**，"
        f"单轮训练耗时由分类头拟合的 {v2['median_epoch_seconds']:.2f} s 变为 "
        f"{v3['median_epoch_seconds']:.2f} s。V2 的骨干前向一次性开销见第 3 节。",
        f"3. V4 更新 **{v4['parameters_trainable']:,}** 个参数，占自身总参数 "
        f"**{v4['parameters_trainable'] / v4['parameters_total']:.2%}**。"
        f"其分配显存峰值为 **{v4['peak_gib']:.3f} GiB**（V3 为 {v3['peak_gib']:.3f} GiB），"
        f"测试 top-1 为 **{v4['top1']['mean']:.2%}**，相对 V3 变化 {gap(v4, v3):+.2f} "
        "个百分点。V4 仍需保存部分中间激活来更新注意力层的 LoRA 参数。",
        "",
        "推理对验证集图像 `" + demo["image"] + "` 调用 PEFT 的 `merge_and_unload` "
        f"后输出 top-3。合并前后 top-3 排序一致，最大置信度差值为 "
        f"{demo['max_score_delta']:.2e}。"
        "[合并前后输出](outputs/analysis/lora_merged_prediction.json)",
        "",
        "| 排名 | 预测品种 | 置信度 |",
        "| ---: | --- | ---: |",
    ]
    for index, item in enumerate(demo["top3"], 1):
        report.append(f"| {index} | {item['breed']} | {item['confidence']:.2%} |")


def main():
    data = read_json(OUTPUTS / "data" / "summary.json")
    all_runs = load_runs()
    summary = summarize_main(all_runs)
    ablations = ablation_tables()
    save_json(OUTPUTS / "results.json", {"main": summary, "ablations": ablations})
    report = ["# Oxford-IIIT Pet 视觉模型微调实验报告", ""]
    add_data(report, data)
    add_setup(report, all_runs)
    add_results(report, summary, all_runs)
    add_ablations(report, ablations)
    add_analysis(report, all_runs)
    add_conclusions(report, summary)
    (ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
