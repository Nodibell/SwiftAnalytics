# SwiftAnalytics

**SwiftAnalytics** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK)**.

---

## 🚀 Core Modules

* **`SwiftDataFrame`**: High-performance columnar data manipulation with zero-copy semantics, built on top of `Apache Arrow`.
* **`SwiftStats`**: Vectorized descriptive statistics, distributions, and hypothesis tests powered by `Accelerate vDSP`.
* **`SwiftPreprocessing`**: Feature scaling, categorical encoding, binning, and preprocessing pipelines (`Pipeline`).
* **`SwiftML`**: Classical estimators (Linear/Logistic Regression, Decision Trees, Random Forests, GBDTs).
* **`SwiftCluster`**: Dimensionality reduction (SVD-based PCA, DBSCAN) and clustering (K-Means).
* **`SwiftOptimize`**: Model validation (K-Fold cross-validation) and parallel hyperparameter optimization (`GridSearchCV`).
* **`SwiftForecast`**: Time series analysis (additive/multiplicative decomposition, Holt-Winters, ARIMA, Kalman filtering).
* **`SwiftNLP`**: Tokenization (Word / subword BPE Tokenizers) and static word embeddings.
* **`SwiftExplain`**: Black-box model explainability using a parallelized `KernelSHAP` implementation.
* **`SwiftLLM`**: Local text generation on GPU using casual transformer-decoder architectures and MLX.
* **`SwiftPrivacy`**: Cryptographic machine learning on encrypted data (Ring-LWE, PNNS) utilizing homomorphic encryption.

---

## 📊 Performance Comparison (Swift vs Python)

The following table presents median execution times on an **Apple Silicon M-series (macOS 15 / arm64)** system, compared directly against popular Python counterparts (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (1M elements) | 0.082 ms | 0.117 ms | 1.43x | 🟢 Swift |
| **StdDev** (1M elements) | 0.473 ms | 0.505 ms | 1.07x | 🟢 Swift |
| **Variance** (1M elements) | 0.453 ms | 0.504 ms | 1.11x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.777 ms | 1.220 ms | 1.57x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 169.470 ms | 18.081 ms | 0.11x | 🔴 Python |
| **Filter rows** (100k rows) | 29.262 ms | 0.532 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.210 ms | 1.572 ms | 0.71x | 🔴 Python |
| **SortBy double column** (100k rows) | 76.899 ms | 6.285 ms | 0.08x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 26.777 ms | 24.064 ms | 0.90x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 4.630 ms | 23.626 ms | 5.10x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 33.453 ms | 30.537 ms | 0.91x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 42.950 ms | 12.228 ms | 0.28x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 2.036 ms | 0.755 ms | 0.37x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.349 ms | 135.124 ms | 21.28x | 🟢 Swift |
| **ARIMA fit** (50k points) | 2.297 ms | 204.394 ms | 89.00x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.388 ms | 206.879 ms | 86.62x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 44.709 ms | 81.172 ms | 1.82x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.431 ms | 0.099 ms | 0.23x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.505 ms | 0.513 ms | 1.02x | 🟢 Swift |
| **LLM Generate** (10 tokens) | 4.924 ms | 3.656 ms | 0.74x | 🔴 Python |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.172 ms | 0.464 ms | 2.69x | 🟢 Swift |
| **RingLWE Encrypt/Decrypt** (vector size=64) | 0.020 ms | 0.286 ms | 14.28x | 🟢 Swift |
| **PNNS Classify** (50 DB vectors, size=64) | 0.231 ms | 2.903 ms | 12.54x | 🟢 Swift |

---

## 🛠 Architectural Highlights & Optimizations

### 1. Transitioning from OOP to Data-Oriented Design (DOD)
Traditional object-oriented trees (where every node is a reference type containing child node pointers) suffer from severe Automatic Reference Counting (ARC) overhead and poor CPU cache locality (L1/L2 cache misses). 
`SwiftAnalytics` resolves this by storing trees as a contiguous flat array of `FlatTreeNode` structures:
* Nodes are allocated next to each other in memory.
* Child navigation is done via array offsets.
* This dramatically increases tree ensemble traversal speeds.

### 2. Intelligent Hardware Routing
We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):
* Branch-heavy algorithms (such as Decision Trees, Random Forests, or spatial DBSCAN search) run strictly on CPU.
* Large tensor operations (Linear/Logistic Regression, K-Means) leverage Apple Silicon GPU execution blocks via `MLXArray` lazy evaluation.

### 3. Optional NaN Validation Bypass
We identified that running $O(N)$ CPU sweeps to check for `NaN` presence in statistical calculations adds significant CPU bottlenecks that limit the speed of underlying `vDSP` functions. In v1.0, we introduced a `checkNaN: Bool = true` default parameter to descriptive stats. When bypassed (set to `false` in benchmarks and hot loops), SwiftStats leverages pure hardware SIMD speeds, outperforming NumPy.

---

## ⚠️ Known Gaps & Limitations

1. **High CSV Parsing Complexity**: The current CSV reader implementation in `SwiftDataFrame` runs with $O(N)$ CPU overhead and does not support streaming parsing of large datasets, resulting in slower reading rates than Pandas. Improving I/O parsing is scheduled for future minor releases.
2. **MLX Sendable Isolation**: `MLXArray` does not conform to `Sendable`. This is handled in the library through strict actor isolation and passing memory ownership tokens (`WiredMemoryTicket`).

---

## 💻 Quick Start

```swift
import SwiftDataFrame
import SwiftStats
import SwiftML

// 1. Load CSV data
let df = try DataFrame.readCSV(contentsOf: csvURL)

// 2. Compute descriptive statistics without NaN checks for maximum speed
let summary = try Stats.describe(df["target"].toDoubles()!, checkNaN: false)
print(summary)

// 3. Train a GPU-accelerated Estimator
let regressor = LinearRegression(device: .gpu)
try await regressor.fit(features: X, targets: y)
let predictions = try await regressor.predict(features: X_test)
```

---

## 📜 License
This project is licensed under the MIT License.
