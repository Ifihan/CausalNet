#!/usr/bin/env python3
# Evaluate CausalNet's pretrained per-subject weights; outputs match phasemer's format.
import argparse
import csv
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Models import CausalNet
from main_train import crop_optical_flow_block, recognition_evaluation

CLASSES = ["Negative", "Positive", "surprise"]

DEFAULT_SUBJECTS = ['sub17', 'sub26', 'sub16', 'sub09', 'sub05', 'sub24', 'sub02', 'sub13', 'sub04', 'sub23',
                'sub11', 'sub12', 'sub08', 'sub14', 'sub03', 'sub19', 'sub01',
                'sub20', 'sub21', 'sub22', 'sub15', 'sub06', 'sub25', 'sub07', '006', '007', '009', '010',
                '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023',
                '024', '026', '028', '030', '032', '031', '033', '034', '035', '036', '037', 's01', 's02',
                's03', 's04', 's05', 's06', 's08', 's09', 's11', 's12', 's13', 's14', 's15', 's18', 's19',
                's20']


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CausalNet's pretrained per-subject weights.")
    p.add_argument("--weights-dir", default="weights", help="Folder with <subject>.pth files.")
    p.add_argument("--subject-list-file", default="", help="Optional JSON with a 'subjects' list to restrict to.")
    p.add_argument("--out-dir", default="results_eval")
    return p.parse_args()


def load_subjects(args):
    if not args.subject_list_file:
        return DEFAULT_SUBJECTS
    with open(args.subject_list_file) as f:
        return json.load(f)["subjects"]


def _plot_confusion(y_true, y_pred, classes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(classes)
    m = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_xticklabels(classes)
    ax.set_yticks(range(k)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (ground truth)")
    thresh = m.max() / 2 if m.max() else 0
    for i in range(k):
        row = m[i].sum()
        for j in range(k):
            cell = f"{m[i, j]}\n({m[i, j] / row:.0%})" if row and i == j else str(m[i, j])
            ax.text(j, i, cell, ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > thresh else "black")
    ax.set_title("CausalNet confusion matrix (pooled held-out test)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    args = parse_args()
    subjects = load_subjects(args)
    gamma = 0.5
    layer1, layer2, layer3 = 3, 3, 9
    main_path = "./datasets/three_norm_u_v_os"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading precomputed optical flow blocks...")
    all_five_parts_optical_flow = crop_optical_flow_block()
    print("Loaded.")

    total_gt, total_pred, rows = [], [], []

    for subject_name in subjects:
        weight_path = os.path.join(args.weights_dir, f"{subject_name}.pth")
        if not os.path.exists(weight_path):
            print(f"[Skipped] Weight not found for subject {subject_name}")
            continue

        print(f"\n===== Testing Subject: {subject_name} =====")

        model = CausalNet(
            image_size=28, patch_size=7, dim=256, heads=3,
            num_hierarchies=3, block_repeats=(layer1, layer2, layer3),
            num_classes=3, gamma=gamma,
        ).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        y_test, four_parts_test = [], []
        subject_path = os.path.join(main_path, subject_name, "u_test")
        if not os.path.exists(subject_path):
            print(f"[Skipped] Test data not found for subject {subject_name}")
            continue

        for n_expression in os.listdir(subject_path):
            for n_img in os.listdir(os.path.join(subject_path, n_expression)):
                imgs = []
                for i in range(4):
                    suffix = f"_{i} .png" if i > 0 else " .png"
                    name = n_img.split(" ")[0] + suffix
                    if name not in all_five_parts_optical_flow:
                        print(f"[Warning] Missing optical flow image: {name}")
                        continue
                    l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[name][0], all_five_parts_optical_flow[name][1]])
                    r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[name][3], all_five_parts_optical_flow[name][4]])
                    imgs.append(cv2.vconcat([l_eye_lips, r_eye_lips]))
                if len(imgs) == 4:
                    y_test.append(int(n_expression))
                    four_parts_test.append(imgs)

        if not four_parts_test:
            print(f"[Skipped] No valid test samples for {subject_name}")
            continue

        y_test_tensor = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test_tensor = torch.Tensor(np.array(four_parts_test)).permute(0, 1, 4, 2, 3)
        test_loader = DataLoader(TensorDataset(four_parts_test_tensor, y_test_tensor), batch_size=batch_size)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.max(model(x), 1)[1]
                for gt, pd_ in zip(y.cpu().tolist(), preds.cpu().tolist()):
                    total_gt.append(gt)
                    total_pred.append(pd_)
                    rows.append([subject_name, gt, pd_])

        uf1, uar = recognition_evaluation(total_gt, total_pred)
        print(f"  [{subject_name}] running pooled UF1={uf1:.4f} UAR={uar:.4f}")

    preds_path = os.path.join(args.out_dir, "preds_causalnet.csv")
    with open(preds_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "y_true", "y_pred"])
        w.writerows(rows)
    print(f"wrote {preds_path} ({len(rows)} rows)")

    uf1, uar = recognition_evaluation(total_gt, total_pred)
    print(f"\nFinal: UF1={uf1:.4f} UAR={uar:.4f}")

    _plot_confusion(total_gt, total_pred, CLASSES, os.path.join(args.out_dir, "confusion_causalnet.png"))


if __name__ == "__main__":
    main()