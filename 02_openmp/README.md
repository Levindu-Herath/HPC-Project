# OpenMP Parallel K-Means Implementation

This is the **shared memory parallel implementation** of the K-Means clustering algorithm using **OpenMP**.
It is developed based on the serial baseline version and parallelizes the most computationally expensive parts of the algorithm.

## Algorithm Overview

The K-Means algorithm works as follows:

1. **Initialization**: Select K initial centroids (using k-means++ or random initialization)
2. **Assignment Step**: Assign each data point to the nearest centroid
3. **Update Step**: Recalculate centroid positions as the mean of assigned points
4. **Repeat**: Continue steps 2–3 until convergence or maximum iterations reached

In this OpenMP version, the heavy loops are **parallelized across multiple CPU threads**.

---

# Parallelization Strategy

The following sections of the algorithm are parallelized using **OpenMP directives**.

### 1. Cluster Assignment (Parallel For)

Each thread processes a subset of points independently.

```c
#pragma omp parallel for reduction(+:changes)
```

Each point calculates its distance to all centroids and selects the nearest cluster.

This is the **most computationally expensive step** and benefits significantly from parallel execution.

---

### 2. Centroid Update

Updating centroids requires **summing all points belonging to each cluster**.

To avoid race conditions:

* Each thread maintains **local sums**
* A **critical section** combines thread results into global centroids

```c
#pragma omp critical
```

---

### 3. WCSS Calculation

The **Within-Cluster Sum of Squares (WCSS)** calculation is parallelized using reduction.

```c
#pragma omp parallel for reduction(+:wcss)
```

Each thread contributes to the global WCSS value safely.

---

# Time Complexity

Per Iteration:

O(N × K × D)

Where:

* **N** = number of data points
* **K** = number of clusters
* **D** = number of dimensions

Parallelization reduces execution time by distributing work across threads.

---

# Space Complexity

O(N × D + K × D)

Additional memory is used for **thread-local centroid sums**.

---

# Compilation

Compile using OpenMP support.

```bash
gcc -O3 -fopenmp -o kmeans_openmp kmeans_openmp.c -lm
```

---

# Usage

```
./kmeans_openmp <input_file> <num_clusters> <max_iterations> <num_threads>
```

---

### Example

Run from the **openmp folder**:

```bash
./kmeans_openmp ../datasets/data_10k.txt 5 100 4
```

This will:

* Load **10,000 data points**
* Cluster into **5 groups**
* Run **100 iterations maximum**
* Use **4 CPU threads**

---

# Output

## Console Output

* Dataset information
* Iteration progress
* Final WCSS value
* Execution time
* Cluster sizes

---

# Performance Metrics

The OpenMP implementation measures:

* **Execution Time**
* **WCSS (Clustering Quality)**
* **Iterations to Convergence**
* **Thread Scalability**

---

# Parallel Performance Evaluation

To evaluate performance, run the program with different thread counts:

```bash
./kmeans_openmp ../datasets/data_10k.txt 5 100 1
./kmeans_openmp ../datasets/data_10k.txt 5 100 2
./kmeans_openmp ../datasets/data_10k.txt 5 100 4
./kmeans_openmp ../datasets/data_10k.txt 5 100 8
```

Measure the execution time for each run.

---

# Speedup Calculation

Speedup is calculated as:

Speedup = Serial Execution Time / Parallel Execution Time

Example table for analysis report:

| Threads | Execution Time (s) | Speedup |
| ------- | ------------------ | ------- |
| 1       | 0.24               | 1.0     |
| 2       | 0.14               | 1.71    |
| 4       | 0.08               | 3.00    |
| 8       | 0.05               | 4.80    |

---

# Accuracy Validation

To verify correctness, compare results with the **serial implementation**.

Use metrics such as:

* **RMSE (Root Mean Square Error)**
* **WCSS comparison**

Parallel and serial outputs should be nearly identical.

---

# Advantages of OpenMP Implementation

* Utilizes **multi-core processors**
* Reduces execution time significantly
* Easy integration with existing serial code
* Efficient for **shared memory systems**

---

# Files

* `kmeans_openmp.c` – OpenMP parallel implementation
* `README.md` – Documentation for OpenMP version

---

# Notes

* OpenMP uses **shared memory parallelism**
* Best performance occurs when **threads ≈ CPU cores**
* Distance calculation and cluster assignment provide the largest speedup
* Reduction and synchronization are required to maintain correctness

---

# Next Steps

After completing the OpenMP implementation:

1. Implement **POSIX Threads version**
2. Implement **MPI distributed version**
3. Implement **Hybrid (MPI + OpenMP) version**
4. Compare performance of all implementations

---

This OpenMP implementation demonstrates how shared-memory parallel programming can significantly accelerate the K-Means clustering algorithm.
