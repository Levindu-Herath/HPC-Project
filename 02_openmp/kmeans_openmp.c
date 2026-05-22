#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

typedef struct {
    double *coords;
    int cluster_id;
} Point;

typedef struct {
    double *coords;
    int count;
} Centroid;

Point *points = NULL;
Centroid *centroids = NULL;
int num_points = 0, num_clusters = 0, dimensions = 0;

void read_data(const char *filename);
void initialize_centroids();
double calculate_distance(double *p1, double *p2);
int assign_clusters_parallel();
void update_centroids_parallel();
double calculate_wcss_parallel();
void cleanup();

void read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { printf("Cannot open file\n"); exit(1); }
    fscanf(fp, "%d %d", &num_points, &dimensions);
    points = malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++) {
        points[i].coords = malloc(dimensions * sizeof(double));
        points[i].cluster_id = -1;
        for (int d = 0; d < dimensions; d++) fscanf(fp, "%lf", &points[i].coords[d]);
    }
    fclose(fp);
}

void initialize_centroids() {
    centroids = malloc(num_clusters * sizeof(Centroid));
    for (int i = 0; i < num_clusters; i++) {
        centroids[i].coords = malloc(dimensions * sizeof(double));
        centroids[i].count = 0;
    }
    int first_idx = rand() % num_points;
    for (int d = 0; d < dimensions; d++) centroids[0].coords[d] = points[first_idx].coords[d];

    for (int k = 1; k < num_clusters; k++) {
        double *distances = malloc(num_points * sizeof(double));
        double sum_distances = 0.0;
        
        #pragma omp parallel for reduction(+ : sum_distances)
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
            if (cumsum >= rand_val) { selected = i; break; }
        }
        for (int d = 0; d < dimensions; d++) centroids[k].coords[d] = points[selected].coords[d];
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

int assign_clusters_parallel() {
    int changes = 0;
    #pragma omp parallel for reduction(+ : changes)
    for (int i = 0; i < num_points; i++) {
        double min_dist = DBL_MAX;
        int nearest = 0;
        for (int k = 0; k < num_clusters; k++) {
            double dist = calculate_distance(points[i].coords, centroids[k].coords);
            if (dist < min_dist) { min_dist = dist; nearest = k; }
        }
        if (points[i].cluster_id != nearest) {
            points[i].cluster_id = nearest;
            changes++;
        }
    }
    return changes;
}

void update_centroids_parallel() {
    for (int k = 0; k < num_clusters; k++) {
        centroids[k].count = 0;
        for (int d = 0; d < dimensions; d++) centroids[k].coords[d] = 0.0;
    }

    #pragma omp parallel
    {
        double local_sum[num_clusters][dimensions];
        int local_count[num_clusters];
        for (int k = 0; k < num_clusters; k++) {
            local_count[k] = 0;
            for (int d = 0; d < dimensions; d++) local_sum[k][d] = 0.0;
        }

        #pragma omp for
        for (int i = 0; i < num_points; i++) {
            int cid = points[i].cluster_id;
            for (int d = 0; d < dimensions; d++) local_sum[cid][d] += points[i].coords[d];
            local_count[cid]++;
        }

        #pragma omp critical
        {
            for (int k = 0; k < num_clusters; k++) {
                for (int d = 0; d < dimensions; d++) centroids[k].coords[d] += local_sum[k][d];
                centroids[k].count += local_count[k];
            }
        }
    }

    for (int k = 0; k < num_clusters; k++) {
        if (centroids[k].count > 0) {
            for (int d = 0; d < dimensions; d++) centroids[k].coords[d] /= centroids[k].count;
        }
    }
}

double calculate_wcss_parallel() {
    double wcss = 0.0;
    #pragma omp parallel for reduction(+ : wcss)
    for (int i = 0; i < num_points; i++) {
        int cid = points[i].cluster_id;
        double dist = calculate_distance(points[i].coords, centroids[cid].coords);
        wcss += dist * dist;
    }
    return wcss;
}

void cleanup() {
    for (int i = 0; i < num_points; i++) free(points[i].coords); free(points);
    for (int k = 0; k < num_clusters; k++) free(centroids[k].coords); free(centroids);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <input_file> <clusters> <iterations> <threads>\n", argv[0]);
        return 1;
    }

    char *input = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int threads = atoi(argv[4]);

    omp_set_num_threads(threads);
    srand(42); 

    read_data(input);

    printf("\n==================================================\n");
    printf("[OpenMP] K-Means Clustering\n");
    printf("==================================================\n");
    printf("Dataset    : %d points, %d dimensions\n", num_points, dimensions);
    printf("Parameters : %d clusters, %d max iterations, %d threads\n", num_clusters, max_iter, threads);
    printf("--------------------------------------------------\n");

    double start = omp_get_wtime();
    initialize_centroids();

    int iter = 0;
    int changes = num_points;
    double final_wcss = 0.0;

    while (iter < max_iter && changes > 0) {
        changes = assign_clusters_parallel();
        update_centroids_parallel();
        iter++;

        if (iter % 10 == 0 || changes == 0) {
            final_wcss = calculate_wcss_parallel();
            double rmse = sqrt(final_wcss / num_points);
            printf("[Iter %3d] WCSS: %15.6f | RMSE: %9.6f | Changes: %6d\n", 
                   iter, final_wcss, rmse, changes);
        }
    }

    double end = omp_get_wtime();
    double final_rmse = sqrt(final_wcss / num_points);

    printf("--------------------------------------------------\n");
    printf("Converged  : %s (%d iterations)\n", (changes == 0 ? "Yes" : "No"), iter);
    printf("Final WCSS : %.6f\n", final_wcss);
    printf("Final RMSE : %.6f\n", final_rmse);
    printf("Time Taken : %.6f seconds\n", end - start);
    printf("==================================================\n\n");

    cleanup();
    return 0;
}