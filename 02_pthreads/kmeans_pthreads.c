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

typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
    int local_changes;
    double **local_centroid_coords; 
    int *local_centroid_counts;     
} ThreadArg;

// Global variables
Point *points = NULL;
Centroid *centroids = NULL;
ThreadArg *global_thread_args = NULL; // Accessible by thread 0 for reduction

int num_points = 0;
int num_clusters = 0;
int dimensions = 0;
int num_threads = 4; 

// Synchronization and loop control variables
pthread_barrier_t barrier;
volatile int global_iteration = 0;
volatile int global_total_changes = 0;
int global_max_iterations = 0;

// Function prototypes
void read_data(const char *filename);
void initialize_centroids();
double calculate_distance(double *p1, double *p2);
void* worker_thread(void* arg);
double calculate_wcss();
void cleanup();

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

double calculate_distance(double *p1, double *p2) {
    double sum = 0.0;
    for (int d = 0; d < dimensions; d++) {
        double diff = p1[d] - p2[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

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

// Persistent Worker Thread Function
void* worker_thread(void* arg) {
    ThreadArg *t_arg = (ThreadArg *)arg;
    int id = t_arg->thread_id;

    while (1) {
        // 1. Check termination conditions
        if (global_iteration >= global_max_iterations || global_total_changes == 0) {
            break;
        }

        // 2. Reset local accumulators for this iteration
        t_arg->local_changes = 0;
        for (int k = 0; k < num_clusters; k++) {
            t_arg->local_centroid_counts[k] = 0;
            for (int d = 0; d < dimensions; d++) {
                t_arg->local_centroid_coords[k][d] = 0.0;
            }
        }

        // 3. Assign points to nearest cluster and accumulate locally
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

            t_arg->local_centroid_counts[nearest_cluster]++;
            for (int d = 0; d < dimensions; d++) {
                t_arg->local_centroid_coords[nearest_cluster][d] += points[i].coords[d];
            }
        }

        // 4. Wait for all threads to finish their assignment phase
        pthread_barrier_wait(&barrier);

        // 5. Thread 0 performs the global reduction and updates condition variables
        if (id == 0) {
            global_total_changes = 0;
            
            // Reset global centroids
            for (int k = 0; k < num_clusters; k++) {
                centroids[k].count = 0;
                for (int d = 0; d < dimensions; d++) centroids[k].coords[d] = 0.0;
            }

            // Reduce data from all thread local arrays
            for (int t = 0; t < num_threads; t++) {
                global_total_changes += global_thread_args[t].local_changes;
                for (int k = 0; k < num_clusters; k++) {
                    centroids[k].count += global_thread_args[t].local_centroid_counts[k];
                    for (int d = 0; d < dimensions; d++) {
                        centroids[k].coords[d] += global_thread_args[t].local_centroid_coords[k][d];
                    }
                }
            }

            // Finalize global centroid means
            for (int k = 0; k < num_clusters; k++) {
                if (centroids[k].count > 0) {
                    for (int d = 0; d < dimensions; d++) {
                        centroids[k].coords[d] /= centroids[k].count;
                    }
                }
            }

            global_iteration++;
            if (global_iteration % 10 == 0 || global_total_changes == 0) {
                double wcss = calculate_wcss();
                printf("Iteration %d: WCSS = %.6f, Changes = %d\n", global_iteration, wcss, global_total_changes);
            }
        }

        // 6. Wait for Thread 0 to finish updating globals before starting the next loop
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

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
    global_max_iterations = atoi(argv[3]);
    num_threads = atoi(argv[4]);

    srand(42);
    
    read_data(input_file);
    initialize_centroids();

    // Initialize the barrier
    pthread_barrier_init(&barrier, NULL, num_threads);

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    global_thread_args = (ThreadArg *)malloc(num_threads * sizeof(ThreadArg));
    
    int chunk_size = num_points / num_threads;
    int remainder = num_points % num_threads;

    for (int i = 0; i < num_threads; i++) {
        global_thread_args[i].thread_id = i;
        global_thread_args[i].start_idx = i * chunk_size + (i < remainder ? i : remainder);
        global_thread_args[i].end_idx = global_thread_args[i].start_idx + chunk_size + (i < remainder ? 1 : 0);
        
        global_thread_args[i].local_centroid_coords = (double **)malloc(num_clusters * sizeof(double *));
        for (int k = 0; k < num_clusters; k++) {
            global_thread_args[i].local_centroid_coords[k] = (double *)malloc(dimensions * sizeof(double));
        }
        global_thread_args[i].local_centroid_counts = (int *)malloc(num_clusters * sizeof(int));
    }

    printf("\nStarting K-Means iterations with %d POSIX threads...\n", num_threads);

    // Initialize loop conditions
    global_iteration = 0;
    global_total_changes = num_points;

    // Use clock_gettime for true wall-clock measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Spawn threads ONCE
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, (void *)&global_thread_args[i]);
    }

    // Wait for all threads to finish all iterations
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate elapsed time in seconds
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("\nConverged after %d iterations\nExecution time: %.6f seconds\n", global_iteration, elapsed_time);

    // Cleanup memory and barrier
    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < num_threads; i++) {
        for (int k = 0; k < num_clusters; k++) {
            free(global_thread_args[i].local_centroid_coords[k]);
        }
        free(global_thread_args[i].local_centroid_coords);
        free(global_thread_args[i].local_centroid_counts);
    }
    free(threads);
    free(global_thread_args);
    cleanup();

    return 0;
}