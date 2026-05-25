#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Global variables for Master Rank
double *global_points = NULL;
int global_num_points = 0;
int dimensions = 0;

void read_data_flat(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { 
        printf("Error: Cannot open file %s\n", filename); 
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    
    // Check if it successfully read the header
    if (fscanf(fp, "%d %d", &global_num_points, &dimensions) != 2) {
        printf("Error: Failed to read dataset header from %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    global_points = (double *)malloc(global_num_points * dimensions * sizeof(double));
    for (int i = 0; i < global_num_points; i++) {
        for (int d = 0; d < dimensions; d++) {
            // Check if it successfully read each data point
            if (fscanf(fp, "%lf", &global_points[i * dimensions + d]) != 1) {
                printf("Error: Failed to read data point at index %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    fclose(fp);
}

void init_centroids(double *points, double *centroids, int num_points, int num_clusters, int dims) {
    int first_idx = rand() % num_points;
    for (int d = 0; d < dims; d++) centroids[0 * dims + d] = points[first_idx * dims + d];

    for (int k = 1; k < num_clusters; k++) {
        double *distances = (double *)malloc(num_points * sizeof(double));
        double sum_distances = 0.0;

        for (int i = 0; i < num_points; i++) {
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++) {
                double dist = 0.0;
                for (int d = 0; d < dims; d++) {
                    double diff = points[i * dims + d] - centroids[j * dims + d];
                    dist += diff * diff;
                }
                dist = sqrt(dist);
                if (dist < min_dist) min_dist = dist;
            }
            distances[i] = min_dist * min_dist;
            sum_distances += distances[i];
        }

        double rand_val = ((double)rand() / RAND_MAX) * sum_distances;
        double cumsum = 0.0;
        int selected = 0;

        for (int i = 0; i < num_points; i++) {
            cumsum += distances[i];
            if (cumsum >= rand_val) { selected = i; break; }
        }

        for (int d = 0; d < dims; d++) centroids[k * dims + d] = points[selected * dims + d];
        free(distances);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    
    // Initialize the MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: mpirun -np <procs> %s <input_file> <num_clusters> <max_iter>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    int num_clusters = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    double *global_centroids = NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Rank 0 Reads Data and Initializes K-Means++
    if (rank == 0) {
        srand(42); 
        read_data_flat(input_file);
        global_centroids = (double *)malloc(num_clusters * dimensions * sizeof(double));
        init_centroids(global_points, global_centroids, global_num_points, num_clusters, dimensions);
        
        printf("\n==================================================\n");
        printf("[MPI] K-Means Clustering (Distributed Memory)\n");
        printf("==================================================\n");
        printf("Dataset    : %d points, %d dimensions\n", global_num_points, dimensions);
        printf("Parameters : %d clusters, %d max iterations, %d MPI Ranks\n", num_clusters, max_iter, size);
        printf("--------------------------------------------------\n");
    }

    // Broadcast Dataset Metadata to all nodes
    MPI_Bcast(&global_num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate centroid memory on worker ranks now that they know the dimensions
    if (rank != 0) {
        global_centroids = (double *)malloc(num_clusters * dimensions * sizeof(double));
    }

    // Broadcast the initial centroids to all nodes
    MPI_Bcast(global_centroids, num_clusters * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition Data using MPI_Scatterv (handles uneven division of points)
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    
    int remainder = global_num_points % size;
    int base_count = global_num_points / size;
    int current_displ = 0;

    for (int i = 0; i < size; i++) {
        int count = base_count + (i < remainder ? 1 : 0);
        sendcounts[i] = count * dimensions;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    // Allocate memory for the local chunk of data on each rank
    int local_num_points = sendcounts[rank] / dimensions;
    double *local_points = (double *)malloc(local_num_points * dimensions * sizeof(double));

    // Scatter the dataset across the network
    MPI_Scatterv(global_points, sendcounts, displs, MPI_DOUBLE, local_points, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Local tracking variables
    int *local_cluster_ids = (int *)malloc(local_num_points * sizeof(int));
    for (int i = 0; i < local_num_points; i++) local_cluster_ids[i] = -1;

    double *local_sums = (double *)malloc(num_clusters * dimensions * sizeof(double));
    int *local_counts = (int *)malloc(num_clusters * sizeof(int));
    
    double *global_sums = (double *)malloc(num_clusters * dimensions * sizeof(double));
    int *global_counts = (int *)malloc(num_clusters * sizeof(int));

    int iter = 0;
    int global_changes = global_num_points;
    double global_wcss = 0.0;

    // Distributed K-Means Loop
    while (iter < max_iter && global_changes > 0) {
        int local_changes = 0;
        
        // Reset local sums and counts
        for (int k = 0; k < num_clusters; k++) {
            local_counts[k] = 0;
            for (int d = 0; d < dimensions; d++) local_sums[k * dimensions + d] = 0.0;
        }

        // Step A: Assign Clusters Locally
        for (int i = 0; i < local_num_points; i++) {
            double min_dist = DBL_MAX;
            int nearest = 0;
            
            for (int k = 0; k < num_clusters; k++) {
                double dist = 0.0;
                for (int d = 0; d < dimensions; d++) {
                    double diff = local_points[i * dimensions + d] - global_centroids[k * dimensions + d];
                    dist += diff * diff;
                }
                dist = sqrt(dist);
                
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = k;
                }
            }
            
            if (local_cluster_ids[i] != nearest) {
                local_cluster_ids[i] = nearest;
                local_changes++;
            }
            
            // Accumulate local sums
            local_counts[nearest]++;
            for (int d = 0; d < dimensions; d++) {
                local_sums[nearest * dimensions + d] += local_points[i * dimensions + d];
            }
        }

        // Step B: Network Reduction (Merge all local math into global variables)
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sums, global_sums, num_clusters * dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, num_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Step C: Update Centroids globally
        for (int k = 0; k < num_clusters; k++) {
            if (global_counts[k] > 0) {
                for (int d = 0; d < dimensions; d++) {
                    global_centroids[k * dimensions + d] = global_sums[k * dimensions + d] / global_counts[k];
                }
            }
        }

        iter++;

        // Calculate WCSS every 10 loops or on convergence
        if (iter % 10 == 0 || global_changes == 0) {
            double local_wcss = 0.0;
            for (int i = 0; i < local_num_points; i++) {
                int cid = local_cluster_ids[i];
                double dist = 0.0;
                for (int d = 0; d < dimensions; d++) {
                    double diff = local_points[i * dimensions + d] - global_centroids[cid * dimensions + d];
                    dist += diff * diff;
                }
                local_wcss += dist; 
            }
            
            MPI_Reduce(&local_wcss, &global_wcss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double rmse = sqrt(global_wcss / global_num_points);
                printf("[Iter %3d] WCSS: %15.6f | RMSE: %9.6f | Changes: %6d\n", iter, global_wcss, rmse, global_changes);
            }
        }
    }

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double final_rmse = sqrt(global_wcss / global_num_points);
        printf("--------------------------------------------------\n");
        printf("Converged  : %s (%d iterations)\n", (global_changes == 0 ? "Yes" : "No"), iter);
        printf("Final WCSS : %.6f\n", global_wcss);
        printf("Final RMSE : %.6f\n", final_rmse);
        printf("Time Taken : %.6f seconds\n", end_time - start_time);
        printf("==================================================\n\n");
    }

    // Cleanup Memory
    free(local_points); free(local_cluster_ids);
    free(local_sums); free(local_counts);
    free(global_sums); free(global_counts);
    free(global_centroids); free(sendcounts); free(displs);
    if (rank == 0) free(global_points);

    MPI_Finalize();
    return 0;
}