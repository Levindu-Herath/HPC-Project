# High Performance Parallel K-Means Clustering

## Implementation Flow and Project Plan

---

## Phase 1: Serial Implementation (Week 1)

**Folder: `01_serial/`**

### Steps:

1. **Understand K-Means Algorithm**
   - Initialization: Select K random centroids
   - Assignment: Assign each point to nearest centroid
   - Update: Recalculate centroid positions
   - Iterate until convergence

2. **Implement Serial Code**
   - Data structure design
   - Distance calculation (Euclidean)
   - Cluster assignment logic
   - Centroid update logic
   - Convergence checking

3. **Test & Validate**
   - Use synthetic dataset (2D/3D points)
   - Verify cluster assignments
   - Measure execution time
   - Save results for comparison

### Deliverables:

- `kmeans_serial.c` - Complete serial implementation
- `data_generator.c` - Dataset generation tool
- `Makefile` - Build system
- Results and timing data

---

## Phase 2: Shared Memory Programming - OpenMP (Week 2)

**Folder: `02_openmp/`**

### Parallelization Strategy:

1. **Distance Calculation Parallelization**
   - Parallelize outer loop over data points
   - Each thread calculates distances for subset of points
   - Use `#pragma omp parallel for`

2. **Cluster Assignment**
   - Parallel reduction for cluster sums
   - Thread-safe centroid updates
   - Critical sections for shared data

3. **Optimizations**
   - Static vs dynamic scheduling
   - Loop collapsing
   - False sharing prevention

### Testing:

- Run with threads: 1, 2, 4, 8, 16
- Measure speedup and efficiency
- Compare accuracy with serial code (RMSE)

---

## Phase 3: Distributed Memory Programming - MPI (Week 3)

**Folder: `03_mpi/`**

### Parallelization Strategy:

1. **Data Distribution**
   - Master process distributes data chunks to workers
   - Each process works on subset of points

2. **Communication Pattern**
   - Broadcast centroids to all processes
   - Local distance calculations
   - Gather partial cluster sums
   - Master updates global centroids

3. **Load Balancing**
   - Equal data distribution
   - Handle uneven workload

### Testing:

- Run with processes: 1, 2, 4, 8
- Measure communication overhead
- Calculate parallel efficiency

---

## Phase 4: Hybrid Programming - MPI + OpenMP (Week 4)

**Folder: `04_hybrid_mpi_openmp/`**

### Strategy:

1. **Two-Level Parallelism**
   - MPI: Distribute data across nodes
   - OpenMP: Thread-level parallelism within each node

2. **Implementation**
   - MPI for inter-node communication
   - OpenMP for intra-node parallelization
   - Optimize thread count per process

### Testing:

- Various combinations: (2 processes × 4 threads), (4 × 2), etc.
- Find optimal configuration
- Compare with pure MPI and pure OpenMP

---

## Phase 5: GPU Programming - CUDA (Week 5)

**Folder: `05_cuda/`**

### Parallelization Strategy:

1. **Kernel Design**
   - Distance calculation kernel (most compute-intensive)
   - Cluster assignment kernel
   - Centroid update with reduction

2. **Memory Management**
   - Transfer data to GPU memory
   - Use shared memory for centroids
   - Optimize memory access patterns

3. **Optimization**
   - Coalesced memory access
   - Occupancy optimization
   - Stream processing

### Testing:

- Different block sizes: 128, 256, 512
- Compare CPU vs GPU performance
- Analyze data transfer overhead

---

## Phase 6: Analysis & Report (Week 6)

### Performance Metrics:

1. **Execution Time**
   - Total time
   - Time per iteration
   - Breakdown by component

2. **Speedup**
   - Speedup = T_serial / T_parallel
   - Plot speedup vs number of processors/threads

3. **Efficiency**
   - Efficiency = Speedup / Number of processors
   - Identify scalability limits

4. **Accuracy**
   - RMSE between parallel and serial results
   - Cluster center differences
   - Convergence iteration count

### Diagrams to Include:

1. **Algorithm Flow Diagram**
   - Serial algorithm steps
   - Parallel decomposition strategy

2. **Performance Graphs**
   - Execution time vs data size
   - Speedup curves
   - Strong scaling & weak scaling
   - Efficiency plots

3. **Architecture Diagrams**
   - MPI process communication pattern
   - OpenMP thread distribution
   - CUDA kernel organization

### Report Structure:

1. Introduction & Problem Statement
2. K-Means Algorithm Description
3. Serial Implementation Details
4. Parallel Implementation Strategies
   - OpenMP approach
   - MPI approach
   - Hybrid approach
   - CUDA approach
5. Performance Analysis
6. Accuracy Validation
7. Conclusions & Future Work

---

## Dataset Specifications

### Synthetic Dataset:

- **Size**: 100K to 10M points
- **Dimensions**: 2D, 3D, or higher
- **Clusters**: 5-10 well-separated clusters
- **Format**: CSV or binary

### Real-World Options:

- UCI Machine Learning Repository
- Kaggle datasets
- Image segmentation data
- Customer clustering data

---

## Build & Run Instructions

### Serial:

```bash
cd 01_serial
make
./kmeans_serial ../datasets/data_100k.txt 5 100
```

### OpenMP:

```bash
cd 02_openmp
make
export OMP_NUM_THREADS=4
./kmeans_openmp ../datasets/data_100k.txt 5 100
```

### MPI:

```bash
cd 03_mpi
make
mpirun -np 4 ./kmeans_mpi ../datasets/data_100k.txt 5 100
```

### Hybrid:

```bash
cd 04_hybrid_mpi_openmp
make
export OMP_NUM_THREADS=4
mpirun -np 2 ./kmeans_hybrid ../datasets/data_100k.txt 5 100
```

### CUDA:

```bash
cd 05_cuda
make
./kmeans_cuda ../datasets/data_100k.txt 5 100
```

---

## Timeline

| Week | Task                  | Deliverable                     |
| ---- | --------------------- | ------------------------------- |
| 1    | Serial implementation | Working code + baseline results |
| 2    | OpenMP implementation | Shared memory parallel version  |
| 3    | MPI implementation    | Distributed memory version      |
| 4    | Hybrid MPI+OpenMP     | Combined approach               |
| 5    | CUDA implementation   | GPU accelerated version         |
| 6    | Testing & Report      | Complete analysis report        |

---

## Key Performance Indicators

- **Speedup > 3x** with 4 threads (OpenMP)
- **Efficiency > 0.7** for small process counts (MPI)
- **GPU speedup > 10x** for large datasets (CUDA)
- **RMSE < 1e-5** (accuracy validation)

---

## References & Resources

1. **CUDA K-Means**: NVIDIA CUDA Samples
2. **MPI Examples**: MPI Tutorial website
3. **OpenMP Guide**: OpenMP official documentation
4. **Papers**: "Parallel K-Means Clustering" research papers
