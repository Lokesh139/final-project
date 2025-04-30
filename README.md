# Traffic Speed Forecasting using Graph Neural Networks

## Overview

This project presents a Graph Neural Network (GNN)-based solution for short-term traffic speed forecasting, leveraging the spatial structure of road networks to model complex traffic dynamics. Using the **METR-LA** dataset, the model demonstrates high accuracy in predicting future traffic speeds, aiding smart city infrastructure and intelligent transport systems.

---

## Key Features

- **Spatiotemporal Modeling** with custom GNN architecture
- **Real-world Dataset**: METR-LA (Los Angeles highway sensors)
- **PyTorch Geometric & Lightning** for modular and scalable development
- **Performance**: MAE = 3.92, RMSE = 7.34 on test data
- **Robust EDA** to inform model architecture and preprocessing
- **Practical Deployment Considerations** for smart cities

---

## Objectives

- Analyze temporal and spatial patterns in traffic data
- Build a GNN that captures both spatial dependencies and short-term traffic dynamics
- Evaluate model accuracy using real-world benchmarks
- Suggest improvements and future directions for production deployment

---

## Dataset

### METR-LA (Main Focus)
- 207 loop detectors in Los Angeles County
- 5-minute interval traffic speed readings
- Complex patterns, missing data, dynamic fluctuations

### PEMS-BAY (For Comparison)
- 325 sensors across the San Francisco Bay Area
- Stable daily congestion patterns

---

## Exploratory Data Analysis Highlights

| Aspect                    | METR-LA Insights                                   |
|--------------------------|----------------------------------------------------|
| Seasonality              | Weak, irregular daily cycles                       |
| Sensor Correlation       | Lower inter-sensor correlation; high variability   |
| Auto-correlation         | Rapid decay; low long-term memory                  |
| Data Gaps                | Frequent missing segments (e.g., full days)        |
| PCA Results              | High dimensionality; complex variance structures   |

---

## Methodology

### Preprocessing
- **Sliding Windows**: 12-timestep input → 1-step (5 min) forecast
- **Z-Score Normalization**: Sensor-specific mean and std
- **Static Graph Construction**: Based on road proximity and adjacency
- **Chronological Split**: Train (70%), Val (20%), Test (10%)

### Model Architecture
- 2-layer GCN with ReLU and Dropout
- Final Fully Connected layer for prediction
- Trained with MSE Loss
- Optimized via Early Stopping

---

## Results

| Metric  | Value  |
|---------|--------|
| MAE     | 3.92   |
| RMSE    | 7.34   |

- **Convergence**: Smooth loss curves
- **Robust Generalization** on unseen traffic patterns
- **Efficient Scaling** to large sensor networks

---

## Tools & Libraries

- Python 3.9+
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- NumPy, Pandas, Matplotlib, Seaborn

---

## Future Work

- Integrate RNNs (LSTM/GRU) for richer temporal modeling
- Use **Graph Attention Networks (GAT)** for dynamic neighbor weighting
- Incorporate external features (weather, incidents)
- Extend to **multi-step forecasting**
- Deploy on edge devices or stream data pipelines

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate on test set
python evaluate.py

# (Optional) Visualize predictions
python visualize_results.py
