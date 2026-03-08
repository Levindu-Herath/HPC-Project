# High Performance Parallel K-Means Clustering

**Course**: EE7218/EC7207 - High Performance Computing  
**Project**: Parallel K-Means Clustering for Large-Scale Data  
**Technologies**: OpenMP, POSIX Threads, MPI, CUDA

---

## 📋 Project Overview

This project implements K-Means clustering algorithm using various parallel programming paradigms and compares their performance against a serial baseline implementation.

### Objectives

1. Implement serial K-Means clustering
2. Parallelize using shared memory (OpenMP)
3. Parallelize using distributed memory (MPI)
4. Implement hybrid approach (MPI + OpenMP)
5. GPU acceleration using CUDA
6. Comprehensive performance analysis

---

## 📁 Project Structure

```
Mini Project/
│
├── PROJECT_IMPLEMENTATION_FLOW.md    # Detailed implementation guide
├── README.md                         # This file
│
├── datasets/                         # Dataset generation and storage
│   ├── data_generator.c             # Synthetic data generator
│   ├── Makefile
│   └── *.txt                        # Generated datasets
│
├── 01_serial/                       # Serial baseline implementation
│   ├── kmeans_serial.c
│   ├── Makefile
│   └── README.md
│
├── 02_openmp/                       # OpenMP parallel implementation
│   └── (To be implemented)
│
├── 03_mpi/                          # MPI distributed implementation
│   └── (To be implemented)
│
├── 04_hybrid_mpi_openmp/           # Hybrid MPI+OpenMP implementation
│   └── (To be implemented)
│
├── 05_cuda/                         # CUDA GPU implementation
│   └── (To be implemented)
│
└── results/                         # Output results and timing data
    ├── serial_results.txt
    ├── serial_timing.txt
    └── ...
```

---

## 🚀 Quick Start

### Step 1: Generate Dataset

```bash
cd datasets
make
./data_generator 10000 5 2 data_10k.txt
./data_generator 100000 8 2 data_100k.txt
./data_generator 1000000 10 2 data_1m.txt
cd ..
```

Or generate all at once:

```bash
cd datasets
make generate_samples
cd ..
```

### Step 2: Build and Run Serial Implementation

```bash
cd 01_serial
make
./kmeans_serial ../datasets/data_10k.txt 5 100
cd ..
```

### Step 3: Check Results

```bash
type results\serial_results.txt
```

---

## 📊 Dataset Specifications

### Synthetic Datasets

| Dataset | Size | Clusters | Dimensions | File          |
| ------- | ---- | -------- | ---------- | ------------- |
| Small   | 10K  | 5        | 2D         | data_10k.txt  |
| Medium  | 100K | 8        | 2D         | data_100k.txt |
| Large   | 1M   | 10       | 2D         | data_1m.txt   |

### Dataset Format

```
<num_points> <dimensions>
x1 y1 [z1 ...]
x2 y2 [z2 ...]
...
```

---

## 🔧 Requirements

### For Serial and OpenMP

- GCC compiler with C99 support
- Make
- Math library (-lm)

### For MPI

- MPI implementation (OpenMPI or MPICH)
- mpicc compiler wrapper

### For CUDA

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- nvcc compiler

---

## 📈 Performance Metrics

### Metrics to Measure

1. **Execution Time**
   - Total runtime
   - Time per iteration
   - Breakdown by component (distance calc, centroid update)

2. **Speedup**
   - Speedup = T_serial / T_parallel
   - Strong scaling (fixed problem size)
   - Weak scaling (scaled problem size)

3. **Efficiency**
   - Efficiency = Speedup / Number of processors
   - Parallel efficiency percentage

4. **Accuracy**
   - RMSE between serial and parallel results
   - Cluster center differences
   - WCSS (Within-Cluster Sum of Squares)

### Expected Performance Goals

- **OpenMP**: 3-4x speedup with 4 threads
- **MPI**: 70%+ efficiency with 4 processes
- **Hybrid**: Best of both worlds
- **CUDA**: 10-50x speedup for large datasets

---

## 📝 Implementation Progress

- [x] Project structure setup
- [x] Dataset generator
- [x] Serial K-Means implementation
- [ ] OpenMP implementation
- [ ] MPI implementation
- [ ] Hybrid MPI+OpenMP implementation
- [ ] CUDA implementation
- [ ] Performance analysis
- [ ] Final report

---

## 🧪 Testing Strategy

### Correctness Validation

1. Use fixed random seed for reproducibility
2. Compare cluster centers across implementations
3. Calculate RMSE between serial and parallel results
4. Visual inspection for 2D datasets

### Performance Testing

1. Run each version with varying:
   - Dataset sizes: 10K, 100K, 500K, 1M, 10M points
   - Thread/process counts: 1, 2, 4, 8, 16
   - Cluster counts: 5, 10, 20, 50
2. Measure execution time (average of 3-5 runs)
3. Record convergence iterations
4. Monitor memory usage

---

## 📚 Key Algorithm Details

### K-Means Algorithm

```
1. Initialize K centroids (using k-means++)
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
3. Return final cluster assignments
```

### Parallelization Opportunities

1. **Distance Calculation** (60-70% of runtime)
   - Independent for each point
   - Highly parallelizable

2. **Cluster Assignment** (20-30% of runtime)
   - Independent per point
   - No data dependencies

3. **Centroid Update** (5-10% of runtime)
   - Requires reduction/aggregation
   - More challenging to parallelize

---

## 📖 Documentation

- [PROJECT_IMPLEMENTATION_FLOW.md](PROJECT_IMPLEMENTATION_FLOW.md) - Detailed phase-by-phase implementation guide
- [01_serial/README.md](01_serial/README.md) - Serial implementation details
- Each folder contains specific README with implementation notes

---

## 🎯 Deliverables

1. ✅ **Serial Code** - Baseline implementation
2. ⏳ **Shared Memory Code** - OpenMP version
3. ⏳ **Distributed Memory Code** - MPI version
4. ⏳ **Hybrid Code** - MPI+OpenMP version
5. ⏳ **GPU Code** - CUDA version (optional but recommended)
6. ⏳ **Analysis Report** - Complete performance analysis

---

## 📊 Analysis Report Contents

The final report should include:

1. **Introduction**
   - Problem statement
   - K-Means algorithm overview
   - Parallelization motivation

2. **Parallel Design**
   - Diagram showing parallel decomposition
   - Communication patterns
   - Load balancing strategy

3. **Implementation Details**
   - Code structure
   - Parallel regions identification
   - Synchronization mechanisms

4. **Results**
   - Performance graphs (execution time, speedup, efficiency)
   - Accuracy validation (RMSE tables)
   - Strong and weak scaling analysis

5. **Discussion**
   - Performance bottlenecks
   - Comparison between approaches
   - Optimal configurations

6. **Conclusion**
   - Summary of findings
   - Lessons learned
   - Future improvements

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Compilation errors with math library  
**Solution**: Make sure to link with `-lm` flag

**Issue**: Dataset file not found  
**Solution**: Use relative paths or check file location

**Issue**: Segmentation fault  
**Solution**: Check array bounds and memory allocation

**Issue**: Different results each run  
**Solution**: Use fixed random seed (already set to 42)

---

## 📧 Support

For questions about:

- K-Means algorithm: See algorithm documentation
- Parallel programming: Consult course materials
- Implementation: Check README files in each folder

---

## 📜 License

This is an academic project for HPC course.

---

## 🙏 Acknowledgments

- Course: EE7218/EC7207 - High Performance Computing
- Various open-source K-Means implementations for reference
- CUDA and MPI documentation

---

**Good luck with your project! 🚀**
