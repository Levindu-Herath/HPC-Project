# Quick Start Guide

Follow these steps to get started with the serial K-Means implementation:

## Step 1: Generate Sample Dataset

```bash
cd datasets
gcc -O2 -o data_generator data_generator.c -lm
./data_generator 10000 5 2 data_10k.txt
cd ..
```

Expected output:

```
=== K-Means Dataset Generator ===

Generating 5 clusters in 2D space...
Cluster 0 center: 2.00 3.00 (std_dev: 2.50)
Cluster 1 center: 22.00 1.00 (std_dev: 2.30)
...

Dataset generated successfully: data_10k.txt
Total points: 10000
Dimensions: 2
True clusters: 5
```

## Step 2: Build Serial K-Means

```bash
cd 01_serial
gcc -O3 -o kmeans_serial kmeans_serial.c -lm
```

## Step 3: Run K-Means Clustering

```bash
./kmeans_serial ../datasets/data_10k.txt 5 100
```

This runs K-Means with:

- 10,000 data points
- 5 clusters
- Maximum 100 iterations

Expected output:

```
=== Serial K-Means Clustering ===

Configuration:
  Input file: ../datasets/data_10k.txt
  Number of clusters (K): 5
  Max iterations: 100

Reading dataset: 10000 points, 2 dimensions
Data loaded successfully!
Centroids initialized using k-means++

Starting K-Means iterations...
Iteration 10: WCSS = 40234.567890, Changes = 45
Iteration 20: WCSS = 39876.543210, Changes = 12
Iteration 25: WCSS = 39850.123456, Changes = 0

=== Results ===
Converged after 25 iterations
Final WCSS: 39850.123456
Execution time: 0.234567 seconds

Cluster Sizes:
  Cluster 0: 2034 points
  Cluster 1: 1987 points
  Cluster 2: 2045 points
  Cluster 3: 1956 points
  Cluster 4: 1978 points

Results written to: ../results/serial_results.txt
```

## Step 4: View Results

```bash
# On Windows
type ..\results\serial_results.txt

# On Linux/Mac
cat ../results/serial_results.txt
```

## Step 5: Generate Larger Datasets for Testing

```bash
cd ../datasets
./data_generator 100000 8 2 data_100k.txt
./data_generator 500000 10 2 data_500k.txt
./data_generator 1000000 10 2 data_1m.txt
cd ../01_serial
```

## Step 6: Test with Larger Dataset

```bash
./kmeans_serial ../datasets/data_100k.txt 8 100
```

## Performance Baseline

Record these metrics for comparison with parallel versions:

- Execution time
- Number of iterations to convergence
- Final WCSS value

## Next Steps

1. ✅ Serial implementation complete
2. Next: Implement OpenMP version in `02_openmp/`
3. Then: Implement MPI version in `03_mpi/`
4. Then: Implement hybrid version in `04_hybrid_mpi_openmp/`
5. Optional: Implement CUDA version in `05_cuda/`
6. Finally: Write analysis report

## Troubleshooting

**Q: Compilation error "undefined reference to 'sqrt'"**  
A: Make sure to link math library with `-lm` flag

**Q: File not found error**  
A: Check that you're in the correct directory and using relative paths correctly

**Q: Results directory not found**  
A: The program will create it automatically, but you can create manually:

```bash
mkdir ..\results  # Windows
mkdir ../results  # Linux/Mac
```

**Q: How do I verify correctness?**  
A:

- Check that all points are assigned to clusters (no -1 cluster IDs)
- Verify cluster sizes are reasonable (roughly balanced)
- WCSS should decrease with iterations
- Use fixed seed (42) for reproducible results

## Understanding the Output

- **WCSS**: Within-Cluster Sum of Squares - lower is better
- **Changes**: Number of points that changed clusters - should decrease
- **Convergence**: When changes = 0 or max iterations reached
- **Cluster Sizes**: Should be roughly balanced for synthetic data

## Tips for Analysis Report

1. Save timing data from multiple runs
2. Test with different dataset sizes
3. Plot WCSS vs iterations
4. Record memory usage
5. Document any issues encountered

---

**You're now ready to start the parallel implementations!** 🎉
