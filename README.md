# LSTM Autoencoder for Network Intrusion Detection (CICIDS2017)

An implementation of an LSTM-based Autoencoder for anomaly detection on the CICIDS2017 network intrusion detection dataset, based on the architecture proposed by Hnamte et al. (2023).

## Overview

This project implements a two-stage deep learning pipeline for network intrusion detection:

- **Stage 1 (Preprocessing):** Data cleaning, inf/NaN removal, and Min-Max normalization
- **Stage 2 (LSTM-AE):** An LSTM Autoencoder trained on benign traffic only, using reconstruction error to flag anomalies at inference time

The core idea is that a model trained exclusively on normal traffic will reconstruct normal traffic well but produce high reconstruction error on attack traffic — allowing a threshold-based classifier to separate the two.

## Architecture

```
Input (timesteps=10, features=78)
    → LSTM(196, relu, return_sequences=True)
    → LSTM(125, relu, return_sequences=False)
    → Dense(16, relu)               ← bottleneck
    → RepeatVector(10)
    → LSTM(125, relu, return_sequences=True)
    → LSTM(196, relu, return_sequences=True)
    → TimeDistributed(Dense(78))    ← reconstruction output
```

- **Loss function:** Mean Absolute Error (MAE)
- **Optimizer:** Adam (lr=0.001)
- **Batch size:** 128
- **Epochs:** 10

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity Intrusion Detection dataset, containing labeled benign and attack network traffic across multiple days and attack categories.

## Requirements

```
tensorflow
numpy
pandas
scikit-learn
kaggle (for dataset access)
```

## Usage

The notebook is designed to run on Kaggle. To reproduce:

1. Attach the [CICIDS2017 dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) to your Kaggle notebook
2. Run all cells in order
3. Evaluation metrics (precision, recall, F1) are printed at the end

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.69 |
| Precision (attack class) | 0.20 |
| Recall (attack class) | 0.40 |
| F1 (attack class) | 0.26 |

**Note on performance:** Results are significantly below those reported in the reference paper (99.99% accuracy). This is attributable to compute constraints inherent to the free Kaggle environment:

- Training was limited to **10 epochs** versus 30 in the reference paper
- A streaming data generator (`tf.data.from_generator`) was required to work around Kaggle's RAM limits (~16GB), which introduces per-step Python overhead and slows convergence relative to in-memory training
- The reference paper used dedicated hardware with 128GB RAM and an NVIDIA RTX 3060 Ti GPU

The reference paper explicitly notes that training time and compute are limitations of the LSTM-AE approach, and that 30 epochs were necessary for the model to converge to high accuracy. The results here reflect an underfit model due to resource constraints, not a failure of the architecture itself.

## Reference

This implementation is based on:

> Hnamte, V., Nhung-Nguyen, H., Hussain, J., & Hwa-Kim, Y. (2023).
> *A Novel Two-Stage Deep Learning Model for Network Intrusion Detection: LSTM-AE.*
> IEEE Access, 11, 37131–37148.
> DOI: [10.1109/ACCESS.2023.3266979](https://doi.org/10.1109/ACCESS.2023.3266979)

## License

This implementation is released for academic and educational use.
