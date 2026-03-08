# Serial K-Means Implementation

This is the baseline serial implementation of the K-Means clustering algorithm.

## Algorithm Overview

The K-Means algorithm works as follows:

1. **Initialization**: Select K initial centroids (using k-means++ for better convergence)
2. **Assignment Step**: Assign each data point to the nearest centroid
3. **Update Step**: Recalculate centroid positions as the mean of assigned points
4. **Repeat**: Continue steps 2-3 until convergence or max iterations reached

## Implementation Details

### Key Functions

- `read_data()`: Loads dataset from file
- `initialize_centroids()`: Uses k-means++ algorithm for smart initialization
- `calculate_distance()`: Computes Euclidean distance between points
- `assign_clusters()`: Assigns each point to nearest centroid (O(N*K*D))
- `update_centroids()`: Updates centroid positions (O(N\*D))
- `calculate_wcss()`: Computes Within-Cluster Sum of Squares for quality metric

### Time Complexity

- Per Iteration: O(N × K × D)
  - N = number of points
  - K = number of clusters
  - D = dimensions

### Space Complexity

- O(N × D + K × D)

## Compilation

```bash
make
```

Or manually:

```bash
gcc -O3 -o kmeans_serial kmeans_serial.c -lm
```

## Usage

```bash
./kmeans_serial <input_file> <num_clusters> <max_iterations>
```

### Example

```bash
./kmeans_serial ../datasets/data_10k.txt 5 100
```

This will:

- Load 10,000 data points
- Cluster into 5 groups
- Run maximum 100 iterations
- Save results to `../results/serial_results.txt`
- Append timing to `../results/serial_timing.txt`

## Output

### Console Output

- Dataset information
- Iteration progress (every 10 iterations)
- Final WCSS (Within-Cluster Sum of Squares)
- Execution time
- Cluster sizes

### File Output

- `serial_results.txt`: Detailed results including centroids and sample assignments
- `serial_timing.txt`: CSV format timing data for performance analysis

## Performance Metrics

The implementation measures:

- **Execution Time**: Total runtime in seconds
- **WCSS**: Quality metric (lower is better)
- **Iterations to Convergence**: Number of iterations needed
- **Cluster Distribution**: Points per cluster

## Sample Run

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
```

## Optimization Opportunities

This serial code serves as the baseline. The next phases will parallelize:

1. **Distance Calculation Loop** (compute-intensive)
   - Most time-consuming part
   - Highly parallelizable

2. **Cluster Assignment Loop**
   - Independent operations per point
   - Good for thread-level parallelism

3. **Centroid Update**
   - Requires reduction operations
   - Challenge: maintain correctness

## Next Steps

After validating this serial implementation:

1. Profile to identify hotspots (distance calculation is typically 60-70% of runtime)
2. Use results as baseline for parallel versions
3. Compare accuracy using RMSE between serial and parallel results
4. Measure speedup and efficiency

## Files

- `kmeans_serial.c`: Main implementation
- `Makefile`: Build configuration
- `README.md`: This file

## Notes

- Uses fixed random seed (42) for reproducibility
- Implements k-means++ initialization for better convergence
- WCSS is used as convergence metric
- Timing uses `clock()` function for CPU time measurement
