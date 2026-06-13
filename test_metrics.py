import csv
import glob
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


MASK_DIRS = {
    9: "./mask_ground_9/",
    7: "./mask_ground_7/",
    5: "./mask_ground_5/",
    3: "./mask_ground_3/",
}

ORIG_MASK_DIR = "data/CORN_2/testB/"
ENH_OUT_DIR = "evaluation/star_result/visualization"
CSV_PATH = "evaluation/star_result/metrics.csv"
IMAGE_SIZE = (384, 384)


def snr_loss_np(y_true, y_pred):
    y_std = np.sqrt(np.var(y_true))
    if y_std == 0:
        return float("inf")
    return 10 * np.log10((np.max(y_pred) ** 2) / (y_std ** 2))


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def get_image_id(enh_path):
    return os.path.basename(enh_path).replace("aug_", "").rsplit(".", 1)[0]


def calculate_snr_values(enhanced_img, base):
    img = cv2.resize(enhanced_img, IMAGE_SIZE)

    orig_mask_path = os.path.join(ORIG_MASK_DIR, base + ".tif")
    orig_m = load_grayscale(orig_mask_path)
    orig_m = cv2.resize(orig_m, IMAGE_SIZE)
    xs, ys = np.nonzero(orig_m)

    target = np.zeros_like(img)
    target[xs, ys] = img[xs, ys]

    snr_values = {}
    for r, mdir in MASK_DIRS.items():
        gt_mask_path = os.path.join(mdir, base + ".jpg")
        gt_mask = load_grayscale(gt_mask_path)
        gt_mask = cv2.resize(gt_mask, IMAGE_SIZE)

        back = img.copy()
        back[gt_mask < 127] = 0
        snr_values[r] = snr_loss_np(back, target)

    return snr_values


def calculate_ssim(enhanced_img, base):
    gt_path = os.path.join(ORIG_MASK_DIR, base + ".tif")
    gt = load_grayscale(gt_path)
    enh = cv2.resize(enhanced_img, gt.shape[::-1])
    ssim_val, _ = compare_ssim(gt, enh, full=True)
    return ssim_val


def calculate_metrics():
    enhanced_paths = sorted(glob.glob(os.path.join(ENH_OUT_DIR, "*.png")))
    rows = []

    for enh_path in enhanced_paths:
        base = get_image_id(enh_path)
        enhanced_img = load_grayscale(enh_path)

        snr_values = calculate_snr_values(enhanced_img, base)
        ssim_value = calculate_ssim(enhanced_img, base)

        row = {
            "image_id": base,
            "ssim": ssim_value,
        }
        for r in sorted(MASK_DIRS.keys(), reverse=True):
            row[f"snr_r{r}"] = snr_values[r]
        rows.append(row)

    return rows


def add_average_row(rows):
    if not rows:
        return None

    metric_columns = [key for key in rows[0].keys() if key != "image_id"]
    average_row = {"image_id": "Average"}
    for column in metric_columns:
        average_row[column] = float(np.mean([row[column] for row in rows]))
    return average_row


def save_results_to_csv(rows, csv_path=CSV_PATH):
    if not rows:
        print("No enhanced images found. CSV was not created.")
        return False

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["image_id", "ssim"] + [
        f"snr_r{r}" for r in sorted(MASK_DIRS.keys(), reverse=True)
    ]

    average_row = add_average_row(rows)
    output_rows = rows + ([average_row] if average_row else [])

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return True


def print_results(rows):
    print("image_id -> SSIM, SNR(dB) for r=9, r=7, r=5, r=3\n")
    for row in rows:
        snr_text = ", ".join(f"{row[f'snr_r{r}']:.3f}" for r in [9, 7, 5, 3])
        print(f"{row['image_id']} -> {row['ssim']:.4f}, {snr_text}")

    average_row = add_average_row(rows)
    if average_row:
        print("\nAverage metrics:")
        print(f"SSIM: {average_row['ssim']:.4f}")
        for r in [9, 7, 5, 3]:
            print(f"r={r}: {average_row[f'snr_r{r}']:.3f} dB")


if __name__ == "__main__":
    metric_rows = calculate_metrics()
    print_results(metric_rows)
    if save_results_to_csv(metric_rows):
        print(f"\nSaved CSV to: {CSV_PATH}")
