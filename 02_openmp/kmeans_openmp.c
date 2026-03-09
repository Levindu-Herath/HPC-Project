/*
 * Parallel K-Means using OpenMP
 * High Performance Computing Mini Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

typedef struct
{
    double *coords;
    int cluster_id;
} Point;

typedef struct
{
    double *coords;
    int count;
} Centroid;

Point *points = NULL;
Centroid *centroids = NULL;

int num_points = 0;
int num_clusters = 0;
int dimensions = 0;

void read_data(const char *filename);
void initialize_centroids();
double calculate_distance(double *p1, double *p2);
int assign_clusters_parallel();
void update_centroids_parallel();
double calculate_wcss_parallel();
void cleanup();

/* ================= READ DATA ================= */
void read_data(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        printf("Cannot open file\n");
        exit(1);
    }

    fscanf(fp, "%d %d", &num_points, &dimensions);

    points = malloc(num_points * sizeof(Point));

    for (int i = 0; i < num_points; i++)
    {
        points[i].coords = malloc(dimensions * sizeof(double));
        points[i].cluster_id = -1;

        for (int d = 0; d < dimensions; d++)
        {
            fscanf(fp, "%lf", &points[i].coords[d]);
        }
    }

    fclose(fp);
}

/* ================= INITIALIZE CENTROIDS ================= */
void initialize_centroids()
{
    centroids = malloc(num_clusters * sizeof(Centroid));

    for (int i = 0; i < num_clusters; i++)
    {
        centroids[i].coords = malloc(dimensions * sizeof(double));
        centroids[i].count = 0;
    }

    for (int k = 0; k < num_clusters; k++)
    {
        int index = rand() % num_points;

        for (int d = 0; d < dimensions; d++)
            centroids[k].coords[d] = points[index].coords[d];
    }
}

/* ================= DISTANCE ================= */
double calculate_distance(double *p1, double *p2)
{
    double sum = 0.0;

    for (int d = 0; d < dimensions; d++)
    {
        double diff = p1[d] - p2[d];
        sum += diff * diff;
    }

    return sqrt(sum);
}

/* ================= ASSIGN CLUSTERS (PARALLEL) ================= */
int assign_clusters_parallel()
{
    int changes = 0;

#pragma omp parallel for reduction(+ : changes)
    for (int i = 0; i < num_points; i++)
    {
        double min_dist = DBL_MAX;
        int nearest = 0;

        for (int k = 0; k < num_clusters; k++)
        {
            double dist = calculate_distance(points[i].coords,
                                             centroids[k].coords);

            if (dist < min_dist)
            {
                min_dist = dist;
                nearest = k;
            }
        }

        if (points[i].cluster_id != nearest)
        {
            points[i].cluster_id = nearest;
            changes++;
        }
    }

    return changes;
}

/* ================= UPDATE CENTROIDS (PARALLEL) ================= */
void update_centroids_parallel()
{
    for (int k = 0; k < num_clusters; k++)
    {
        centroids[k].count = 0;
        for (int d = 0; d < dimensions; d++)
            centroids[k].coords[d] = 0.0;
    }

#pragma omp parallel
    {
        double **local_sum = malloc(num_clusters * sizeof(double *));
        int *local_count = calloc(num_clusters, sizeof(int));

        for (int k = 0; k < num_clusters; k++)
        {
            local_sum[k] = calloc(dimensions, sizeof(double));
        }

#pragma omp for
        for (int i = 0; i < num_points; i++)
        {
            int cid = points[i].cluster_id;

            for (int d = 0; d < dimensions; d++)
                local_sum[cid][d] += points[i].coords[d];

            local_count[cid]++;
        }

#pragma omp critical
        {
            for (int k = 0; k < num_clusters; k++)
            {
                for (int d = 0; d < dimensions; d++)
                    centroids[k].coords[d] += local_sum[k][d];

                centroids[k].count += local_count[k];
            }
        }

        for (int k = 0; k < num_clusters; k++)
            free(local_sum[k]);

        free(local_sum);
        free(local_count);
    }

    for (int k = 0; k < num_clusters; k++)
    {
        if (centroids[k].count > 0)
        {
            for (int d = 0; d < dimensions; d++)
                centroids[k].coords[d] /= centroids[k].count;
        }
    }
}

/* ================= WCSS (PARALLEL) ================= */
double calculate_wcss_parallel()
{
    double wcss = 0.0;

#pragma omp parallel for reduction(+ : wcss)
    for (int i = 0; i < num_points; i++)
    {
        int cid = points[i].cluster_id;

        double dist = calculate_distance(points[i].coords,
                                         centroids[cid].coords);

        wcss += dist * dist;
    }

    return wcss;
}

/* ================= CLEANUP ================= */
void cleanup()
{
    for (int i = 0; i < num_points; i++)
        free(points[i].coords);

    free(points);

    for (int k = 0; k < num_clusters; k++)
        free(centroids[k].coords);

    free(centroids);
}

/* ================= MAIN ================= */
int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Usage: %s <input_file> <clusters> <iterations> <threads>\n", argv[0]);
        return 1;
    }

    char *input = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int threads = atoi(argv[4]);

    omp_set_num_threads(threads);

    srand(42);

    double start = omp_get_wtime();

    read_data(input);
    initialize_centroids();

    int iter = 0;
    int changes = num_points;

    while (iter < max_iter && changes > 0)
    {
        changes = assign_clusters_parallel();
        update_centroids_parallel();
        iter++;
    }

    double wcss = calculate_wcss_parallel();

    double end = omp_get_wtime();

    printf("Iterations: %d\n", iter);
    printf("WCSS: %f\n", wcss);
    printf("Execution Time: %f seconds\n", end - start);

    cleanup();

    return 0;
}