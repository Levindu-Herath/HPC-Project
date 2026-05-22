#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

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
int assign_clusters();
void update_centroids();
double calculate_wcss();
void cleanup();

void read_data(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: Cannot open file %s\n", filename); exit(1); }
    fscanf(fp, "%d %d", &num_points, &dimensions);
    points = (Point *)malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++) {
        points[i].coords = (double *)malloc(dimensions * sizeof(double));
        points[i].cluster_id = -1;
        for (int d = 0; d < dimensions; d++) fscanf(fp, "%lf", &points[i].coords[d]);
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
    for (int d = 0; d < dimensions; d++) centroids[0].coords[d] = points[first_idx].coords[d];

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

int assign_clusters() {
    int changes = 0;
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

void update_centroids() {
    for (int k = 0; k < num_clusters; k++) {
        centroids[k].count = 0;
        for (int d = 0; d < dimensions; d++) centroids[k].coords[d] = 0.0;
    }
    for (int i = 0; i < num_points; i++) {
        int cid = points[i].cluster_id;
        if (cid >= 0) {
            for (int d = 0; d < dimensions; d++) centroids[cid].coords[d] += points[i].coords[d];
            centroids[cid].count++;
        }
    }
    for (int k = 0; k < num_clusters; k++) {
        if (centroids[k].count > 0) {
            for (int d = 0; d < dimensions; d++) centroids[k].coords[d] /= centroids[k].count;
        }
    }
}

double calculate_wcss() {
    double wcss = 0.0;
    for (int i = 0; i < num_points; i++) {
        int cid = points[i].cluster_id;
        if (cid >= 0) {
            double dist = calculate_distance(points[i].coords, centroids[cid].coords);
            wcss += dist * dist;
        }
    }
    return wcss;
}

void cleanup() {
    if (points) { for (int i = 0; i < num_points; i++) free(points[i].coords); free(points); }
    if (centroids) { for (int k = 0; k < num_clusters; k++) free(centroids[k].coords); free(centroids); }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input_file> <num_clusters> <max_iterations>\n", argv[0]);
        return 1;
    }

    const char *input = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    srand(42); 
    read_data(input);

    printf("\n==================================================\n");
    printf("[Serial] K-Means Clustering\n");
    printf("==================================================\n");
    printf("Dataset    : %d points, %d dimensions\n", num_points, dimensions);
    printf("Parameters : %d clusters, %d max iterations\n", num_clusters, max_iter);
    printf("--------------------------------------------------\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    initialize_centroids();

    int iter = 0;
    int changes = num_points;
    double final_wcss = 0.0;

    while (iter < max_iter && changes > 0) {
        changes = assign_clusters();
        update_centroids();
        iter++;
        
        if (iter % 10 == 0 || changes == 0) {
            final_wcss = calculate_wcss();
            double rmse = sqrt(final_wcss / num_points);
            printf("[Iter %3d] WCSS: %15.6f | RMSE: %9.6f | Changes: %6d\n", 
                   iter, final_wcss, rmse, changes);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double final_rmse = sqrt(final_wcss / num_points);

    printf("--------------------------------------------------\n");
    printf("Converged  : %s (%d iterations)\n", (changes == 0 ? "Yes" : "No"), iter);
    printf("Final WCSS : %.6f\n", final_wcss);
    printf("Final RMSE : %.6f\n", final_rmse);
    printf("Time Taken : %.6f seconds\n", elapsed);
    printf("==================================================\n\n");

    cleanup();
    return 0;
}