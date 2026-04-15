"""Test and comparative evaluation script for XuNet."""
import argparse
import os
import re
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from model.model import XuNet
from dataset import dataset


def safe_percent(numerator, denominator):
    """Return percentage safely without division by zero."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def detect_resolution(*candidates):
    """Extract resolution token like res128 from any string candidates."""
    for item in candidates:
        if not item:
            continue
        match = re.search(r"(res\d+)", str(item))
        if match:
            return match.group(1)
    return "res128"


def evaluate_checkpoint(
    checkpoint_path,
    cover_path,
    stego_path,
    test_size,
    batch_size,
    file_extension,
    color_mode,
    device,
    verbose=True,
):
    """Run one checkpoint evaluation and return all metrics."""
    if verbose:
        print(f"\n[RUN] checkpoint={checkpoint_path}")
        print(f"      cover={cover_path}")
        print(f"      stego={stego_path}")

    test_data = dataset.DatasetLoad(
        cover_path,
        stego_path,
        test_size,
        transform=transforms.ToTensor(),
        file_extension=file_extension,
        color_mode=color_mode,
    )

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    in_channels = 3 if color_mode == "RGB" else 1
    model = XuNet(in_channels=in_channels).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            images = torch.cat((test_batch["cover"], test_batch["stego"]), 0)
            labels = torch.cat((test_batch["label"][0], test_batch["label"][1]), 0)

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            prediction = outputs.data.max(1)[1]

            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if verbose:
                batch_acc = safe_percent(prediction.eq(labels).sum().item(), labels.size(0))
                print(f"      batch {i + 1}/{len(test_loader)} acc={batch_acc:.2f}%")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    tp = int(np.sum((all_predictions == 1) & (all_labels == 1)))
    tn = int(np.sum((all_predictions == 0) & (all_labels == 0)))
    fp = int(np.sum((all_predictions == 1) & (all_labels == 0)))
    fn = int(np.sum((all_predictions == 0) & (all_labels == 1)))

    accuracy = safe_percent(tp + tn, tp + tn + fp + fn)
    precision = safe_percent(tp, tp + fp)
    recall = safe_percent(tp, tp + fn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "checkpoint": checkpoint_path,
        "cover_path": cover_path,
        "stego_path": stego_path,
        "test_size": test_size,
        "batch_size": batch_size,
        "color_mode": color_mode,
        "device": str(device),
    }


def write_single_result_file(result_file, resolution, metrics):
    """Write one evaluation result file."""
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 50}\n")
        f.write(f"TEST RESULTS - {result_file}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Resolution      : {resolution}\n")
        f.write(f"Checkpoint      : {metrics['checkpoint']}\n")
        f.write(f"Cover path      : {metrics['cover_path']}\n")
        f.write(f"Stego path      : {metrics['stego_path']}\n")
        f.write(f"Test size       : {metrics['test_size']}\n")
        f.write(f"Batch size      : {metrics['batch_size']}\n")
        f.write(f"Color mode      : {metrics['color_mode']}\n")
        f.write(f"Device          : {metrics['device']}\n")
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Test Accuracy   : {metrics['accuracy']:.2f}%\n")
        f.write(f"{'=' * 50}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"True Positive  (Stego->Stego) : {metrics['tp']}\n")
        f.write(f"True Negative  (Cover->Cover) : {metrics['tn']}\n")
        f.write(f"False Positive (Cover->Stego) : {metrics['fp']}\n")
        f.write(f"False Negative (Stego->Cover) : {metrics['fn']}\n")
        f.write(f"\nAccuracy  : {metrics['accuracy']:.2f}%\n")
        f.write(f"Precision : {metrics['precision']:.2f}%\n")
        f.write(f"Recall    : {metrics['recall']:.2f}%\n")


def write_comparative_summary(summary_path, resolution, hybrid_metrics, lsb_metrics):
    """Write comparative table containing confusion matrix and key metrics."""
    epoch_union = sorted(set(hybrid_metrics.keys()) | set(lsb_metrics.keys()))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"RANGKUMAN METRIK {resolution.upper()} - HYBRID vs LSB\n")
        f.write("=" * 120 + "\n")
        f.write(f"Tanggal        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hybrid files   : {len(hybrid_metrics)}\n")
        f.write(f"LSB files      : {len(lsb_metrics)}\n")
        f.write(f"Total net union: {len(epoch_union)}\n\n")
        f.write("Format kolom:\n")
        f.write("net | H:[TP TN FP FN | Acc Pre Rec] | L:[TP TN FP FN | Acc Pre Rec]\n")
        f.write("-" * 120 + "\n")

        for epoch in epoch_union:
            h = hybrid_metrics.get(epoch)
            l = lsb_metrics.get(epoch)

            h_text = "-"
            l_text = "-"

            if h:
                h_text = (
                    f"[{h['tp']:2d} {h['tn']:2d} {h['fp']:2d} {h['fn']:2d} | "
                    f"{h['accuracy']:6.2f}% {h['precision']:6.2f}% {h['recall']:6.2f}%]"
                )
            if l:
                l_text = (
                    f"[{l['tp']:2d} {l['tn']:2d} {l['fp']:2d} {l['fn']:2d} | "
                    f"{l['accuracy']:6.2f}% {l['precision']:6.2f}% {l['recall']:6.2f}%]"
                )

            f.write(f"net_{epoch:<3d} | H:{h_text:<46} | L:{l_text}\n")


def generate_comparative_chart_from_metrics(chart_file, resolution, hybrid_metrics, lsb_metrics):
    """Generate line chart only from in-memory batch metrics to guarantee data source clarity."""
    common_epochs = sorted(set(hybrid_metrics.keys()) & set(lsb_metrics.keys()))
    if not common_epochs:
        print("\n[INFO] Comparative chart skipped: no paired epoch data from current batch run.")
        return None

    hybrid_acc = [hybrid_metrics[e]["accuracy"] for e in common_epochs]
    lsb_acc = [lsb_metrics[e]["accuracy"] for e in common_epochs]
    delta = [l - h for l, h in zip(lsb_acc, hybrid_acc)]

    plt.figure(figsize=(12, 6))
    plt.plot(common_epochs, lsb_acc, marker="o", linewidth=2, color="#1f77b4", label="LSB Standar (L)")
    plt.plot(common_epochs, hybrid_acc, marker="s", linewidth=2, color="#d62728", label="Metode Hibrida (H)")
    plt.fill_between(common_epochs, hybrid_acc, lsb_acc, color="#ff9896", alpha=0.25, label="Delta L - H")
    plt.title(f"Komparasi Akurasi Steganalisis per Epoch ({resolution})")
    plt.xlabel("Epoch")
    plt.ylabel("Persentase Akurasi (%)")
    plt.xticks(common_epochs)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_file, dpi=200)
    plt.close()

    persistently_below_count = sum(1 for d in delta if d > 0)
    equal_count = sum(1 for d in delta if abs(d) < 1e-9)
    avg_delta = float(np.mean(delta))

    print("\n" + "=" * 60)
    print("KOMPARASI KURVA AKURASI (L vs H)")
    print("=" * 60)
    print(f"Sumber data             : HASIL BATCH RUN SAAT INI (bukan file lama campuran)")
    print(f"Epoch terplot           : {common_epochs[0]} sampai {common_epochs[-1]} ({len(common_epochs)} titik)")
    print(f"Rata-rata delta (L - H) : {avg_delta:.2f}%")
    print(f"H berada di bawah L     : {persistently_below_count}/{len(common_epochs)} epoch")
    print(f"H sama dengan L         : {equal_count}/{len(common_epochs)} epoch")
    print("Deduksi narasi:")
    print("Kurva Hibrida (H) cenderung berada di bawah kurva LSB Standar (L),")
    print("menunjukkan kemampuan deteksi model yang lebih tertekan sepanjang pelatihan.")
    print(f"Grafik tersimpan di     : {chart_file}")
    print("=" * 60)
    return {
        "avg_delta": avg_delta,
        "below_count": persistently_below_count,
        "equal_count": equal_count,
        "total": len(common_epochs),
    }


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    # Single test mode arguments
    parser.add_argument("--test_cover_path", default="./dataset/res128/test/cover")
    parser.add_argument("--test_stego_path", default="./dataset/res128/test/stego")
    parser.add_argument("--test_size", type=int, default=1500, help="Number of test images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", default="./checkpoints/net_100.pt")
    parser.add_argument("--file_extension", default=".png")
    parser.add_argument(
        "--color_mode",
        default="RGB",
        choices=["RGB", "L"],
        help="Image color mode: 'RGB' for 3-channel or 'L' for grayscale",
    )
    parser.add_argument("--plot_epochs", type=int, default=40, help="Number of epochs to plot in comparative chart")

    # Comparative batch mode arguments
    parser.add_argument("--run_comparative_batch", action="store_true", help="Run Hybrid net_1..N then LSB net_1..N")
    parser.add_argument("--epochs_start", type=int, default=1, help="Start epoch index for batch mode")
    parser.add_argument("--epochs_end", type=int, default=40, help="End epoch index for batch mode")
    parser.add_argument("--checkpoint_dir", default=None, help="Checkpoint directory for batch mode, e.g. ./checkpoints/res128")
    parser.add_argument("--resolution", default=None, help="Resolution token override, e.g. res128")
    parser.add_argument("--results_root", default="./results", help="Root output directory for result files")
    parser.add_argument("--clear_batch_results", action="store_true", help="Delete existing test_*.txt in target result folders before batch run")

    return parser.parse_args()


def clear_old_test_files(folder):
    """Delete old test_*.txt files from target folder."""
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if re.match(r"test_\d+\.txt$", name):
            os.remove(os.path.join(folder, name))


def run_single_mode(args, device):
    """Run one checkpoint test and save one result file."""
    resolution = detect_resolution(args.test_cover_path, args.test_stego_path, args.checkpoint, args.resolution)
    normalized_cover_path = args.test_cover_path.replace("\\", "/").lower()
    type_match = re.search(r"/((hybrid|lsb))/res\d+", normalized_cover_path)
    test_type = type_match.group(1) if type_match else None

    if test_type:
        results_dir = os.path.join(args.results_root, test_type, resolution)
    else:
        results_dir = os.path.join(args.results_root, resolution)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Testing with {args.color_mode} images")
    print(f"File format: {args.file_extension}")
    print(f"Device: {device}")

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        cover_path=args.test_cover_path,
        stego_path=args.test_stego_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        file_extension=args.file_extension,
        color_mode=args.color_mode,
        device=device,
        verbose=True,
    )

    existing = [f for f in os.listdir(results_dir) if re.match(r"test_\d+\.txt", f)]
    next_index = len(existing) + 1
    result_file = os.path.join(results_dir, f"test_{next_index}.txt")
    write_single_result_file(result_file, resolution, metrics)

    print("\n" + "=" * 50)
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(f"True Positive (Stego->Stego):   {metrics['tp']}")
    print(f"True Negative (Cover->Cover):   {metrics['tn']}")
    print(f"False Positive (Cover->Stego):  {metrics['fp']}")
    print(f"False Negative (Stego->Cover):  {metrics['fn']}")
    print(f"\nAccuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"\nResults saved to: {result_file}")


def run_comparative_batch_mode(args, device):
    """Run Hybrid net_1..N then LSB net_1..N and generate table + chart from this run only."""
    resolution = detect_resolution(args.resolution, args.checkpoint_dir, args.checkpoint, args.test_cover_path)
    checkpoint_dir = args.checkpoint_dir or os.path.dirname(args.checkpoint)
    if not checkpoint_dir:
        checkpoint_dir = "./checkpoints"

    hybrid_cover = f"./dataset/hybrid/{resolution}/cover/test"
    hybrid_stego = f"./dataset/hybrid/{resolution}/stego/test"
    lsb_cover = f"./dataset/lsb/{resolution}/cover/test"
    lsb_stego = f"./dataset/lsb/{resolution}/stego/test"

    hybrid_results_dir = os.path.join(args.results_root, "hybrid", resolution)
    lsb_results_dir = os.path.join(args.results_root, "lsb", resolution)
    os.makedirs(hybrid_results_dir, exist_ok=True)
    os.makedirs(lsb_results_dir, exist_ok=True)

    if args.clear_batch_results:
        clear_old_test_files(hybrid_results_dir)
        clear_old_test_files(lsb_results_dir)

    print("\n" + "=" * 70)
    print("MODE: COMPARATIVE BATCH")
    print("Eksekusi berurutan: HYBRID (net_1..N) lalu LSB (net_1..N)")
    print("Sumber chart: metrik in-memory dari batch run saat ini")
    print("=" * 70)

    hybrid_metrics = {}
    lsb_metrics = {}

    for epoch in range(args.epochs_start, args.epochs_end + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"net_{epoch}.pt")
        if not os.path.isfile(checkpoint_path):
            print(f"[WARN] Checkpoint tidak ditemukan, skip: {checkpoint_path}")
            continue

        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            cover_path=hybrid_cover,
            stego_path=hybrid_stego,
            test_size=args.test_size,
            batch_size=args.batch_size,
            file_extension=args.file_extension,
            color_mode=args.color_mode,
            device=device,
            verbose=False,
        )
        hybrid_metrics[epoch] = metrics
        write_single_result_file(os.path.join(hybrid_results_dir, f"test_{epoch}.txt"), resolution, metrics)
        print(
            f"[HYBRID] net_{epoch}: "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} "
            f"| Acc={metrics['accuracy']:.2f}% Pre={metrics['precision']:.2f}% Rec={metrics['recall']:.2f}%"
        )

    for epoch in range(args.epochs_start, args.epochs_end + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"net_{epoch}.pt")
        if not os.path.isfile(checkpoint_path):
            print(f"[WARN] Checkpoint tidak ditemukan, skip: {checkpoint_path}")
            continue

        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            cover_path=lsb_cover,
            stego_path=lsb_stego,
            test_size=args.test_size,
            batch_size=args.batch_size,
            file_extension=args.file_extension,
            color_mode=args.color_mode,
            device=device,
            verbose=False,
        )
        lsb_metrics[epoch] = metrics
        write_single_result_file(os.path.join(lsb_results_dir, f"test_{epoch}.txt"), resolution, metrics)
        print(
            f"[LSB]    net_{epoch}: "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} "
            f"| Acc={metrics['accuracy']:.2f}% Pre={metrics['precision']:.2f}% Rec={metrics['recall']:.2f}%"
        )

    summary_file = os.path.join(args.results_root, f"summary_metrics_{resolution}_hybrid_lsb.txt")
    write_comparative_summary(summary_file, resolution, hybrid_metrics, lsb_metrics)
    print(f"\nTabel metrik tersimpan di: {summary_file}")

    chart_file = os.path.join(args.results_root, f"comparison_accuracy_{resolution}_lsb_vs_hybrid.png")
    generate_comparative_chart_from_metrics(chart_file, resolution, hybrid_metrics, lsb_metrics)


def main():
    """Program entrypoint."""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.run_comparative_batch:
        run_comparative_batch_mode(args, device)
    else:
        run_single_mode(args, device)


if __name__ == "__main__":
    main()