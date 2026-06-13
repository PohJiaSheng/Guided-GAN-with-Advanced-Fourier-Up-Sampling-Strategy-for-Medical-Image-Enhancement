# FU-GAN: Fourier Up-Sampling Generative Adversarial Network

## Abstract

While recent image enhancement frameworks have improved by injecting high-quality
cues for guidance purposes, the up-sampling process can still directly affect
the edge fidelity and artifact propagation. However, it is still usually treated as a
standard implementation setting. This study reviews the up-sampling process as a
significant reconstruction bottleneck for image enhancement and hence proposes
Fourier Up-Sampling Generative Adversarial Network (FU-GAN), a guided-GAN
framework with the integration of a learnable Fourier-domain up sampling
strategy. The proposed approach has been evaluated on widely used CCM datasets,
CORN-2, where the experimental results show the proposed FU-GAN outperforms
other methods in terms of both enhancement quality and downstream segmentation
performance. The findings in this research illustrate that the up-sampling
process can be the key to overcome bottleneck for image enhancement, where the
proposed Fourier up-sampling technique is a practical mechanism that able to
improve both enhancement quality and practical usability in downstream task.

## Environment Setup

The dependency file targets Python 3.8 and CUDA 11.8.

```bash
conda create -n fu-gan python=3.8 -y
conda activate fu-gan
pip install -r requirements.txt
```

If you are using a different CUDA version or CPU-only environment, install the
matching PyTorch build first, then install the remaining dependencies from
`requirements.txt`.

## Dataset Preparation

Place the CORN-2 dataset under `data/CORN_2` using the following structure:

```text
data/CORN_2/
|-- trainA/   # Low-quality training images
|-- trainB/   # High-quality/reference training images
|-- testA/    # Low-quality testing images
`-- testB/    # High-quality/reference testing images
```

The default configuration in `configs/rnw_star.yaml` reads from
`./data/CORN_2`.

## Model Training

```bash
python train_model.py 
```

## Model Testing

Run enhancement on the test images:

```bash
python test_model.py 
```

## Quantitative Metrics

`test_metrics.py` calculates SSIM and SNR values, then saves the results to CSV.
SNR calculation requires background masks generated from the reference test
images. Run `generate_mask.py` before calculating metrics:

```bash
python generate_mask.py
```

This creates the required mask folders:

```text
mask_ground_9/
mask_ground_7/
mask_ground_5/
mask_ground_3/
```

After testing the model and generating masks, calculate metrics:

```bash
python test_metrics.py
```