# Analysis Report Template

## High Performance Parallel K-Means Clustering

**Student Name:** [Your Name]  
**Course:** EE7218/EC7207 - High Performance Computing  
**Date:** [Submission Date]

---

## 1. Introduction

### 1.1 Problem Statement

K-Means clustering is a widely-used unsupervised machine learning algorithm for grouping data points into K clusters. The algorithm is computationally intensive for large datasets, making it an excellent candidate for parallelization.

**Challenge:** For N points, K clusters, and D dimensions, each iteration requires O(N×K×D) operations, which becomes prohibitive for large-scale data.

### 1.2 Project Objectives

- Implement serial K-Means as baseline
- Develop parallel versions using OpenMP, MPI, and CUDA
- Analyze and compare performance
- Validate accuracy of parallel implementations

### 1.3 Dataset Description

[Describe the datasets you used]

- Dataset sizes: [e.g., 10K, 100K, 1M points]
- Dimensions: [e.g., 2D, 3D]
- Number of clusters: [e.g., 5, 8, 10]
- Data distribution: [synthetic with Gaussian distribution]

---

## 2. K-Means Algorithm Overview

### 2.1 Algorithm Description

```
Algorithm: K-Means Clustering
Input: Dataset X with N points, Number of clusters K
Output: Cluster assignments and centroids

1. Initialize K centroids (using k-means++)
2. Repeat until convergence:
   a. Assignment Step: For each point, assign to nearest centroid
   b. Update Step: Recalculate centroids as mean of assigned points
3. Return final clusters
```

### 2.2 Time Complexity Analysis

- **Per Iteration:** O(N × K × D)
  - Distance calculation: O(N × K × D)
  - Centroid update: O(N × D)
- **Total:** O(I × N × K × D), where I = number of iterations

### 2.3 Hotspot Identification

[Profile your serial code and identify bottlenecks]

Example breakdown:

- Distance calculation: 65%
- Cluster assignment: 25%
- Centroid update: 8%
- Other operations: 2%

---

## 3. Parallel Implementation Strategies

### 3.1 Diagram: Parallel Decomposition

[Include a diagram showing how you parallelized the algorithm]

```
Example structure:

Serial:                 OpenMP:                MPI:
┌─────────────┐        ┌─────────────┐       ┌─────────────┐
│ All Points  │        │ Thread 0    │       │ Process 0   │
│             │   -->  │ Thread 1    │  -->  │ Process 1   │
│             │        │ Thread 2    │       │ Process 2   │
│             │        │ Thread 3    │       │ Process 3   │
└─────────────┘        └─────────────┘       └─────────────┘
      │                      │                      │
      v                      v                      v
 Sequential           Shared Memory          Distributed
 Processing           Parallelism            Memory + Comms
```

### 3.2 OpenMP Implementation

**Parallelization Strategy:**

- Parallelize distance calculation loop
- Use reduction for centroid updates
- Critical sections for thread-safe operations

**Key Code Snippet:**

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < num_points; i++) {
    // Find nearest centroid
    for (int k = 0; k < num_clusters; k++) {
        double dist = calculate_distance(points[i], centroids[k]);
        // Update assignment
    }
}
```

**Challenges & Solutions:**

- Challenge: False sharing in centroid updates
- Solution: Use thread-local buffers and reduction

### 3.3 MPI Implementation

**Parallelization Strategy:**

- Master-worker pattern
- Distribute data points across processes
- Broadcast centroids, gather partial results

**Communication Pattern:**

```
Master Process:
1. Broadcast centroids to all workers
2. Scatter data points
3. Gather cluster assignments
4. Update global centroids

Worker Processes:
1. Receive centroids
2. Process local points
3. Send results back to master
```

**Challenges & Solutions:**

- Challenge: Communication overhead
- Solution: Minimize broadcasts, use non-blocking communications where possible

### 3.4 Hybrid MPI+OpenMP

**Strategy:**

- MPI for inter-node parallelism
- OpenMP for intra-node parallelism
- Two-level parallel decomposition

**Configuration:**
[Describe your setup, e.g., 4 MPI processes × 4 OpenMP threads per process]

### 3.5 CUDA Implementation

**Parallelization Strategy:**

- Distance calculation kernel (one thread per point)
- Reduction kernel for centroid updates
- Host-device memory transfers

**Kernel Design:**

```cuda
__global__ void assign_clusters_kernel(Point *points, Centroid *centroids, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Calculate distances and assign to nearest cluster
    }
}
```

---

## 4. Accuracy Validation

### 4.1 Correctness Verification

**Method:** Compare cluster centroids between serial and parallel implementations using RMSE.

**RMSE Formula:**
$$RMSE = \sqrt{\frac{1}{K \times D} \sum_{k=1}^{K} \sum_{d=1}^{D} (C_{serial}[k][d] - C_{parallel}[k][d])^2}$$

### 4.2 Results

| Implementation | Dataset | K   | RMSE    | Match Serial? |
| -------------- | ------- | --- | ------- | ------------- |
| Serial         | 100K    | 8   | -       | Baseline      |
| OpenMP         | 100K    | 8   | [Value] | ✓             |
| MPI            | 100K    | 8   | [Value] | ✓             |
| Hybrid         | 100K    | 8   | [Value] | ✓             |
| CUDA           | 100K    | 8   | [Value] | ✓             |

[Fill in your actual RMSE values]

**Acceptable Threshold:** RMSE < 1e-5

### 4.3 WCSS Comparison

| Implementation | WCSS Value | % Difference from Serial |
| -------------- | ---------- | ------------------------ |
| Serial         | [Value]    | 0%                       |
| OpenMP         | [Value]    | [%]                      |
| MPI            | [Value]    | [%]                      |
| Hybrid         | [Value]    | [%]                      |
| CUDA           | [Value]    | [%]                      |

---

## 5. Performance Analysis

### 5.1 Experimental Setup

**Hardware:**

- CPU: [e.g., Intel Core i7, 8 cores]
- RAM: [e.g., 16GB]
- GPU: [e.g., NVIDIA GTX 1080, if used]
- OS: [e.g., Windows 11 / Ubuntu 20.04]

**Software:**

- Compiler: [e.g., GCC 11.2]
- MPI Implementation: [e.g., OpenMPI 4.1]
- CUDA Version: [e.g., 11.5, if used]

**Test Configuration:**

- Datasets: 10K, 100K, 500K, 1M points
- Clusters: K = 5, 8, 10
- Iterations: Fixed at 100 (or until convergence)
- Runs: Average of 5 runs per configuration

### 5.2 Execution Time Results

**Table: Execution Time (seconds) for 100K points, K=8**

| Threads/Processes | Serial | OpenMP | MPI  | Hybrid | CUDA |
| ----------------- | ------ | ------ | ---- | ------ | ---- |
| 1                 | [T1]   | [T1]   | [T1] | [T1]   | -    |
| 2                 | -      | [T2]   | [T2] | [T2]   | -    |
| 4                 | -      | [T4]   | [T4] | [T4]   | -    |
| 8                 | -      | [T8]   | [T8] | [T8]   | -    |
| GPU               | -      | -      | -    | -      | [TG] |

[Fill in your measurements]

### 5.3 Speedup Analysis

**Speedup Formula:**
$$Speedup = \frac{T_{serial}}{T_{parallel}}$$

**Graph 1: Speedup vs Number of Threads/Processes**

```
[Include a line graph showing speedup for different implementations]
X-axis: Number of threads/processes (1, 2, 4, 8, 16)
Y-axis: Speedup
Lines: OpenMP, MPI, Hybrid
```

[Create and insert your graph here]

**Table: Speedup**

| Threads/Processes | OpenMP | MPI  | Hybrid | CUDA |
| ----------------- | ------ | ---- | ------ | ---- |
| 1                 | 1.0x   | 1.0x | 1.0x   | -    |
| 2                 | [S2]   | [S2] | [S2]   | -    |
| 4                 | [S4]   | [S4] | [S4]   | -    |
| 8                 | [S8]   | [S8] | [S8]   | -    |
| GPU               | -      | -    | -      | [SG] |

### 5.4 Efficiency Analysis

**Efficiency Formula:**
$$Efficiency = \frac{Speedup}{Number\ of\ Processors} \times 100\%$$

**Graph 2: Efficiency vs Number of Processors**

```
[Include a line graph]
X-axis: Number of processors
Y-axis: Efficiency (%)
Ideal line: 100% efficiency
```

**Table: Parallel Efficiency**

| Threads/Processes | OpenMP | MPI   | Hybrid |
| ----------------- | ------ | ----- | ------ |
| 2                 | [E2]%  | [E2]% | [E2]%  |
| 4                 | [E4]%  | [E4]% | [E4]%  |
| 8                 | [E8]%  | [E8]% | [E8]%  |

**Analysis:**

- Above 80%: Excellent scaling
- 60-80%: Good scaling
- Below 60%: Poor scaling (investigate bottlenecks)

### 5.5 Strong Scaling Analysis

**Fixed Problem Size: 1M points, K=10**

[Graph showing execution time decreasing as processors increase]

### 5.6 Weak Scaling Analysis

**Fixed Work per Processor: 100K points per processor**

| Processors | Dataset Size | Time (s) | Efficiency |
| ---------- | ------------ | -------- | ---------- |
| 1          | 100K         | [T1]     | 100%       |
| 2          | 200K         | [T2]     | [E2]%      |
| 4          | 400K         | [T4]     | [E4]%      |
| 8          | 800K         | [T8]     | [E8]%      |

**Ideal Weak Scaling:** Time remains constant

### 5.7 Breakdown Analysis

**Time Breakdown for 100K points with 4 threads/processes:**

| Component          | Serial | OpenMP | MPI | Speedup |
| ------------------ | ------ | ------ | --- | ------- |
| Distance Calc      | [T]    | [T]    | [T] | [S]     |
| Cluster Assignment | [T]    | [T]    | [T] | [S]     |
| Centroid Update    | [T]    | [T]    | [T] | [S]     |
| Communication      | -      | -      | [T] | -       |
| **Total**          | [T]    | [T]    | [T] | [S]     |

---

## 6. Comparison and Discussion

### 6.1 Performance Comparison

**Best Performer by Scenario:**

| Scenario              | Best Implementation | Reason |
| --------------------- | ------------------- | ------ |
| Small data (<10K)     | [Name]              | [Why]  |
| Medium data (100K-1M) | [Name]              | [Why]  |
| Large data (>1M)      | [Name]              | [Why]  |
| Limited cores (2-4)   | [Name]              | [Why]  |
| Many cores (16+)      | [Name]              | [Why]  |
| Distributed system    | [Name]              | [Why]  |

### 6.2 Bottleneck Analysis

**Identified Bottlenecks:**

1. **OpenMP:**
   - Memory bandwidth saturation with many threads
   - False sharing in centroid updates
   - NUMA effects on multi-socket systems

2. **MPI:**
   - Communication overhead (centroid broadcast)
   - Load imbalance with uneven cluster sizes
   - Startup latency for small datasets

3. **Hybrid:**
   - Optimal thread-to-process ratio difficult to find
   - Over-subscription issues
   - Complexity of tuning two levels

4. **CUDA:**
   - Data transfer overhead (CPU ↔ GPU)
   - Kernel launch overhead
   - Memory coalescing issues

### 6.3 Amdahl's Law Analysis

**Theoretical Maximum Speedup:**

Given serial fraction $f_s$ and parallel fraction $f_p = 1 - f_s$:

$$Speedup_{max} = \frac{1}{f_s + \frac{f_p}{N}}$$

**Your Measurements:**

- Serial fraction estimate: [e.g., 5% = 0.05]
- Parallel fraction: [e.g., 95% = 0.95]
- Theoretical max speedup (8 cores): [Calculate]
- Actual speedup achieved: [Your result]
- Efficiency: [Actual/Theoretical × 100%]

### 6.4 Lessons Learned

**What Worked Well:**

- [List successes]
- [What performed better than expected]

**Challenges Faced:**

- [List difficulties]
- [What didn't work as expected]

**Performance Insights:**

- [Key findings]
- [Surprising results]

---

## 7. Graphs and Visualizations

### 7.1 Required Graphs

1. **Execution Time vs Dataset Size**
   - All implementations on same graph
   - Log-log scale recommended

2. **Speedup vs Number of Processors**
   - Include ideal linear speedup line
   - Compare OpenMP, MPI, Hybrid

3. **Efficiency vs Number of Processors**
   - Show degradation with more processors

4. **Strong Scaling**
   - Fixed problem size
   - Increasing processors

5. **Weak Scaling**
   - Fixed work per processor
   - Should ideally stay flat

6. **Component Breakdown**
   - Stacked bar chart
   - Show time spent in each phase

### 7.2 Sample Visualization

```
Example: Speedup Comparison

    Speedup
      16 ┤                              ╭─ Ideal
      14 ┤                           ╭──
      12 ┤                        ╭──
      10 ┤    ╭─────────────╭────       ← OpenMP
       8 ┤  ╭─              ╭────       ← MPI
       6 ┤╭─              ╭─            ← Hybrid
       4 ┼               ╭─
       2 ┤             ╭─
       0 └────┬────┬────┬────┬────
            1    2    4    8   16
                Processors

Key Observations:
- All implementations show good speedup up to 4 cores
- OpenMP scales better than MPI for shared memory
- MPI has communication overhead visible
```

---

## 8. Conclusions

### 8.1 Summary of Findings

[Summarize your key results]

Example structure:

- Serial implementation achieved [X] seconds for 1M points
- OpenMP provided up to [Y]x speedup with [Z] threads
- MPI achieved [W]x speedup with [V] processes
- Hybrid implementation achieved best of [best speedup]
- CUDA provided [GPU speedup]x speedup for large datasets
- All parallel versions maintained accuracy (RMSE < threshold)

### 8.2 Recommendations

**For Different Scenarios:**

1. **Small Datasets (<100K points):**
   - Recommendation: [Serial or OpenMP with few threads]
   - Reason: [Communication/thread overhead dominates]

2. **Medium Datasets (100K-1M points):**
   - Recommendation: [OpenMP or Hybrid]
   - Reason: [Good balance of speedup and complexity]

3. **Large Datasets (>1M points):**
   - Recommendation: [CUDA or Hybrid MPI+OpenMP]
   - Reason: [GPU excels, or distributed needed for memory]

### 8.3 Future Improvements

**Potential Enhancements:**

1. Dynamic load balancing for uneven clusters
2. Asynchronous communication in MPI
3. GPU stream processing for overlapping computation
4. Approximate methods for very large datasets
5. Mini-batch K-Means for streaming data

**Further Optimizations:**

- SIMD vectorization
- Cache-aware data layouts
- Alternative distance metrics
- Parallel initialization improvements

---

## 9. References

1. Lloyd, S. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory.

2. Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"

3. OpenMP Architecture Review Board. "OpenMP Application Programming Interface". [https://www.openmp.org/](https://www.openmp.org/)

4. MPI Forum. "MPI: A Message-Passing Interface Standard". [https://www.mpi-forum.org/](https://www.mpi-forum.org/)

5. NVIDIA Corporation. "CUDA C Programming Guide". [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

6. Course lecture notes and materials (EE7218/EC7207)

7. [Add any other references you used]

---

## Appendices

### Appendix A: Complete Code Listings

[Include key sections or link to GitHub repository]

### Appendix B: Build and Execution Instructions

[Complete instructions for reproducing your results]

### Appendix C: Raw Performance Data

[Include all timing measurements in tables]

### Appendix D: Hardware Specifications

[Detailed specs of system used for testing]

---

**End of Report**
