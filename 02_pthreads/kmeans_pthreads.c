#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    double *coords;
    int cluster_id;
} Point;

typedef struct {
    double *coords;
    int count;
} Centroid;

// Thread argument structure
typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
    int local_changes;
    double **local_centroid_coords; // Thread-local sums
    int *local_centroid_counts;     // Thread-local counts
} ThreadArg;

// Global variables
Point *points = NULL;
Centroid *centroids = NULL;
int num_points = 0;
int num_clusters = 0;
int dimensions = 0;
int num_threads = 4; // Default thread count

// Function prototypes
void read_data(const char *filename);
void initialize_centroids();
double calculate_distance(double *p1, double *p2);
void* worker_thread(void* arg);
double calculate_wcss();
void write_results(const char *output_file);
void cleanup();

// Read dataset from file (Same as serial)
void read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }
    fscanf(fp, "%d %d", &num_points, &dimensions);
    printf("Reading dataset: %d points, %d dimensions\n", num_points, dimensions);

    points = (Point *)malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++) {
        points[i].coords = (double *)malloc(dimensions * sizeof(double));
        points[i].cluster_id = -1;
    }

    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dimensions; d++) {
            if (fscanf(fp, "%lf", &points[i].coords[d]) != 1) {
                fprintf(stderr, "Error reading data\n");
                exit(1);
            }
        }
    }
    fclose(fp);
}

// Initialize centroids (Same as serial)
void initialize_centroids() {
    centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid));
    for (int i = 0; i < num_clusters; i++) {
        centroids[i].coords = (double *)malloc(dimensions * sizeof(double));
        centroids[i].count = 0;
    }

    int first_idx = rand() % num_points;
    for (int d = 0; d < dimensions; d++) {
        centroids[0].coords[d] = points[first_idx].coords[d];
    }

    for (int k = 1; k < num_clusters; k++) {
        double *distances = (double *)malloc(num_points * sizeof(double));
        double sum_distances = 0.0;

        for (int i = 0; i < num_points; i++) {
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++) {
                double dist = calculate_distance(points[i].coords, centroids[j].coords);
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
            if (cumsum >= rand_val) {
                selected = i;
                break;
            }
        }

        for (int d = 0; d < dimensions; d++) {
            centroids[k].coords[d] = points[selected].coords[d];
        }
        free(distances);
    }
}

// Calculate Euclidean distance
double calculate_distance(double *p1, double *p2) {
    double sum = 0.0;
    for (int d = 0; d < dimensions; d++) {
        double diff = p1[d] - p2[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// POSIX Worker Thread Function
void* worker_thread(void* arg) {
    ThreadArg *t_arg = (ThreadArg *)arg;
    t_arg->local_changes = 0;

    // Reset local centroid accumulators
    for (int k = 0; k < num_clusters; k++) {
        t_arg->local_centroid_counts[k] = 0;
        for (int d = 0; d < dimensions; d++) {
            t_arg->local_centroid_coords[k][d] = 0.0;
        }
    }

    // Assign points to nearest cluster and accumulate local sums
    for (int i = t_arg->start_idx; i < t_arg->end_idx; i++) {
        double min_distance = DBL_MAX;
        int nearest_cluster = 0;

        for (int k = 0; k < num_clusters; k++) {
            double distance = calculate_distance(points[i].coords, centroids[k].coords);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_cluster = k;
            }
        }

        if (points[i].cluster_id != nearest_cluster) {
            points[i].cluster_id = nearest_cluster;
            t_arg->local_changes++;
        }

        // Add to local thread sums
        t_arg->local_centroid_counts[nearest_cluster]++;
        for (int d = 0; d < dimensions; d++) {
            t_arg->local_centroid_coords[nearest_cluster][d] += points[i].coords[d];
        }
    }
    return NULL;
}

// Calculate WCSS
double calculate_wcss() {
    double wcss = 0.0;
    for (int i = 0; i < num_points; i++) {
        int cluster = points[i].cluster_id;
        if (cluster >= 0) {
            double dist = calculate_distance(points[i].coords, centroids[cluster].coords);
            wcss += dist * dist;
        }
    }
    return wcss;
}

// Cleanup memory
void cleanup() {
    if (points) {
        for (int i = 0; i < num_points; i++) free(points[i].coords);
        free(points);
    }
    if (centroids) {
        for (int k = 0; k < num_clusters; k++) free(centroids[k].coords);
        free(centroids);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <input_file> <num_clusters> <max_iterations> <num_threads>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iterations = atoi(argv[3]);
    num_threads = atoi(argv[4]);

    srand(42);
    clock_t start_time = clock();

    read_data(input_file);
    initialize_centroids();

    // Prepare Thread Arguments and Structures
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadArg *thread_args = (ThreadArg *)malloc(num_threads * sizeof(ThreadArg));
    
    int chunk_size = num_points / num_threads;
    int remainder = num_points % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].start_idx = i * chunk_size + (i < remainder ? i : remainder);
        thread_args[i].end_idx = thread_args[i].start_idx + chunk_size + (i < remainder ? 1 : 0);
        
        thread_args[i].local_centroid_coords = (double **)malloc(num_clusters * sizeof(double *));
        for (int k = 0; k < num_clusters; k++) {
            thread_args[i].local_centroid_coords[k] = (double *)malloc(dimensions * sizeof(double));
        }
        thread_args[i].local_centroid_counts = (int *)malloc(num_clusters * sizeof(int));
    }

    int iteration = 0;
    int total_changes = num_points;

    printf("\nStarting K-Means iterations with %d POSIX threads...\n", num_threads);

    while (iteration < max_iterations && total_changes > 0) {
        total_changes = 0;

        // 1. Reset global centroids
        for (int k = 0; k < num_clusters; k++) {
            for (int d = 0; d < dimensions; d++) centroids[k].coords[d] = 0.0;
            centroids[k].count = 0;
        }

        // 2. Spawn threads for assignment and local summation
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, worker_thread, (void *)&thread_args[i]);
        }

        // 3. Join threads and reduce (combine) results
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
            total_changes += thread_args[i].local_changes;

            for (int k = 0; k < num_clusters; k++) {
                centroids[k].count += thread_args[i].local_centroid_counts[k];
                for (int d = 0; d < dimensions; d++) {
                    centroids[k].coords[d] += thread_args[i].local_centroid_coords[k][d];
                }
            }
        }

        // 4. Finalize global centroid means
        for (int k = 0; k < num_clusters; k++) {
            if (centroids[k].count > 0) {
                for (int d = 0; d < dimensions; d++) {
                    centroids[k].coords[d] /= centroids[k].count;
                }
            }
        }

        double wcss = calculate_wcss();
        iteration++;
        if (iteration % 10 == 0 || total_changes == 0) {
            printf("Iteration %d: WCSS = %.6f, Changes = %d\n", iteration, wcss, total_changes);
        }
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nConverged after %d iterations\nExecution time: %.6f seconds\n", iteration, elapsed_time);

    // Cleanup POSIX specific memory
    for (int i = 0; i < num_threads; i++) {
        for (int k = 0; k < num_clusters; k++) {
            free(thread_args[i].local_centroid_coords[k]);
        }
        free(thread_args[i].local_centroid_coords);
        free(thread_args[i].local_centroid_counts);
    }
    free(threads);
    free(thread_args);
    cleanup();

    return 0;
}