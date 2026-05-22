#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIMENSIONS 10

typedef struct
{
    double coords[MAX_DIMENSIONS];
    int cluster_id;
} Point;

typedef struct
{
    double coords[MAX_DIMENSIONS];
    int count;
} Centroid;

Point *h_points = NULL;
Centroid *h_centroids = NULL;

Point *d_points = NULL;
Centroid *d_centroids = NULL;

int num_points = 0;
int num_clusters = 0;
int dimensions = 0;

void read_data(const char *filename);
void initialize_centroids();
void cleanup();
double calculate_wcss();

__device__ double calculate_distance(double *p1, double *p2, int dims)
{
    double sum = 0.0;

    for (int d = 0; d < dims; d++)
    {
        double diff = p1[d] - p2[d];
        sum += diff * diff;
    }

    return sqrt(sum);
}

__global__ void assign_clusters_kernel(Point *points,
                                       Centroid *centroids,
                                       int num_points,
                                       int num_clusters,
                                       int dimensions,
                                       int *changes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points)
        return;

    double min_dist = DBL_MAX;
    int nearest_cluster = 0;

    for (int k = 0; k < num_clusters; k++)
    {
        double dist = calculate_distance(points[idx].coords,
                                         centroids[k].coords,
                                         dimensions);

        if (dist < min_dist)
        {
            min_dist = dist;
            nearest_cluster = k;
        }
    }

    if (points[idx].cluster_id != nearest_cluster)
    {
        atomicAdd(changes, 1);
        points[idx].cluster_id = nearest_cluster;
    }
}

__global__ void reset_centroids_kernel(Centroid *centroids,
                                       int num_clusters,
                                       int dimensions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_clusters)
        return;

    centroids[idx].count = 0;

    for (int d = 0; d < dimensions; d++)
    {
        centroids[idx].coords[d] = 0.0;
    }
}

__global__ void update_centroids_kernel(Point *points,
                                        Centroid *centroids,
                                        int num_points,
                                        int dimensions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points)
        return;

    int cid = points[idx].cluster_id;

    atomicAdd(&centroids[cid].count, 1);

    for (int d = 0; d < dimensions; d++)
    {
        atomicAdd(&centroids[cid].coords[d], points[idx].coords[d]);
    }
}

__global__ void finalize_centroids_kernel(Centroid *centroids,
                                          int num_clusters,
                                          int dimensions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_clusters)
        return;

    if (centroids[idx].count > 0)
    {
        for (int d = 0; d < dimensions; d++)
        {
            centroids[idx].coords[d] /= centroids[idx].count;
        }
    }
}

void read_data(const char *filename)
{
    FILE *fp = fopen(filename, "r");

    if (!fp)
    {
        printf("Cannot open file\n");
        exit(1);
    }

    fscanf(fp, "%d %d", &num_points, &dimensions);

    if (dimensions > MAX_DIMENSIONS)
    {
        printf("Dimensions exceed MAX_DIMENSIONS\n");
        exit(1);
    }

    h_points = (Point *)malloc(num_points * sizeof(Point));

    for (int i = 0; i < num_points; i++)
    {
        h_points[i].cluster_id = -1;

        for (int d = 0; d < dimensions; d++)
        {
            fscanf(fp, "%lf", &h_points[i].coords[d]);
        }
    }

    fclose(fp);

    printf("Dataset Loaded: %d points, %d dimensions\n",
           num_points,
           dimensions);
}

void initialize_centroids()
{
    h_centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid));

    for (int k = 0; k < num_clusters; k++)
    {
        int idx = rand() % num_points;

        h_centroids[k].count = 0;

        for (int d = 0; d < dimensions; d++)
        {
            h_centroids[k].coords[d] = h_points[idx].coords[d];
        }
    }
}

double calculate_wcss()
{
    double wcss = 0.0;

    for (int i = 0; i < num_points; i++)
    {
        int cid = h_points[i].cluster_id;

        double sum = 0.0;

        for (int d = 0; d < dimensions; d++)
        {
            double diff = h_points[i].coords[d] - h_centroids[cid].coords[d];
            sum += diff * diff;
        }

        wcss += sum;
    }

    return wcss;
}

void cleanup()
{
    if (h_points)
        free(h_points);

    if (h_centroids)
        free(h_centroids);

    cudaFree(d_points);
    cudaFree(d_centroids);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <input_file> <clusters> <iterations>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iterations = atoi(argv[3]);

    srand(42);

    read_data(input_file);
    initialize_centroids();

    cudaMalloc((void **)&d_points, num_points * sizeof(Point));
    cudaMalloc((void **)&d_centroids, num_clusters * sizeof(Centroid));

    cudaMemcpy(d_points,
               h_points,
               num_points * sizeof(Point),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_centroids,
               h_centroids,
               num_clusters * sizeof(Centroid),
               cudaMemcpyHostToDevice);

    int *d_changes;
    cudaMalloc((void **)&d_changes, sizeof(int));

    int blockSize = 256;
    int pointBlocks = (num_points + blockSize - 1) / blockSize;
    int centroidBlocks = (num_clusters + blockSize - 1) / blockSize;

    printf("\nStarting CUDA K-Means...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int iteration = 0;

    while (iteration < max_iterations)
    {
        cudaMemset(d_changes, 0, sizeof(int));

        assign_clusters_kernel<<<pointBlocks, blockSize>>>(d_points,
                                                           d_centroids,
                                                           num_points,
                                                           num_clusters,
                                                           dimensions,
                                                           d_changes);

        reset_centroids_kernel<<<centroidBlocks, blockSize>>>(d_centroids,
                                                              num_clusters,
                                                              dimensions);

        update_centroids_kernel<<<pointBlocks, blockSize>>>(d_points,
                                                            d_centroids,
                                                            num_points,
                                                            dimensions);

        finalize_centroids_kernel<<<centroidBlocks, blockSize>>>(d_centroids,
                                                                 num_clusters,
                                                                 dimensions);

        iteration++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_points,
               d_points,
               num_points * sizeof(Point),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_centroids,
               d_centroids,
               num_clusters * sizeof(Centroid),
               cudaMemcpyDeviceToHost);

    double wcss = calculate_wcss();

    printf("\n=== CUDA Results ===\n");
    printf("Iterations: %d\n", iteration);
    printf("WCSS: %f\n", wcss);
    printf("Execution Time: %f seconds\n", milliseconds / 1000.0);

    printf("\nCluster Sizes:\n");

    for (int k = 0; k < num_clusters; k++)
    {
        printf("Cluster %d: %d points\n",
               k,
               h_centroids[k].count);
    }

    FILE *fp = fopen("../results/cuda_timing.txt", "a");

    if (fp)
    {
        fprintf(fp,
                "%d,%d,%d,%d,%f\n",
                num_points,
                dimensions,
                num_clusters,
                iteration,
                milliseconds / 1000.0);

        fclose(fp);
    }

    cleanup();

    printf("\n=== Done ===\n");

    return 0;
}
