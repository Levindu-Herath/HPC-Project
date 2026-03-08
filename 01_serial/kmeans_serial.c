// * Serial K-Means Clustering Implementation
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

typedef struct
{
    double *coords; // Point coordinates
    int cluster_id; // Assigned cluster ID
} Point;

typedef struct
{
    double *coords; // Centroid coordinates
    int count;      // Number of points in cluster
} Centroid;

// Global variables
Point *points = NULL;
Centroid *centroids = NULL;
int num_points = 0;
int num_clusters = 0;
int dimensions = 0;

// Function prototypes
void read_data(const char *filename);
void initialize_centroids();
double calculate_distance(double *p1, double *p2);
int assign_clusters();
void update_centroids();
double calculate_wcss();
void write_results(const char *output_file);
void cleanup();

// Read dataset from file
void read_data(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Read header
    fscanf(fp, "%d %d", &num_points, &dimensions);
    printf("Reading dataset: %d points, %d dimensions\n", num_points, dimensions);

    // Allocate memory for points
    points = (Point *)malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++)
    {
        points[i].coords = (double *)malloc(dimensions * sizeof(double));
        points[i].cluster_id = -1;
    }

    // Read data points
    for (int i = 0; i < num_points; i++)
    {
        for (int d = 0; d < dimensions; d++)
        {
            if (fscanf(fp, "%lf", &points[i].coords[d]) != 1)
            {
                fprintf(stderr, "Error reading data at point %d, dimension %d\n", i, d);
                exit(1);
            }
        }
    }

    fclose(fp);
    printf("Data loaded successfully!\n");
}

// Initialize centroids using k-means++ method
void initialize_centroids()
{
    centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid));
    for (int i = 0; i < num_clusters; i++)
    {
        centroids[i].coords = (double *)malloc(dimensions * sizeof(double));
        centroids[i].count = 0;
    }

    // Choose first centroid randomly
    int first_idx = rand() % num_points;
    for (int d = 0; d < dimensions; d++)
    {
        centroids[0].coords[d] = points[first_idx].coords[d];
    }

    // Choose remaining centroids using k-means++
    for (int k = 1; k < num_clusters; k++)
    {
        double *distances = (double *)malloc(num_points * sizeof(double));
        double sum_distances = 0.0;

        // Calculate minimum distance to existing centroids
        for (int i = 0; i < num_points; i++)
        {
            double min_dist = DBL_MAX;
            for (int j = 0; j < k; j++)
            {
                double dist = calculate_distance(points[i].coords, centroids[j].coords);
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }
            distances[i] = min_dist * min_dist;
            sum_distances += distances[i];
        }

        // Choose next centroid with probability proportional to distance squared
        double rand_val = ((double)rand() / RAND_MAX) * sum_distances;
        double cumsum = 0.0;
        int selected = 0;

        for (int i = 0; i < num_points; i++)
        {
            cumsum += distances[i];
            if (cumsum >= rand_val)
            {
                selected = i;
                break;
            }
        }

        for (int d = 0; d < dimensions; d++)
        {
            centroids[k].coords[d] = points[selected].coords[d];
        }

        free(distances);
    }

    printf("Centroids initialized using k-means++\n");
}

// Calculate Euclidean distance between two points
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

// Assign each point to nearest centroid
// Returns number of points that changed clusters
int assign_clusters()
{
    int changes = 0;

    for (int i = 0; i < num_points; i++)
    {
        double min_distance = DBL_MAX;
        int nearest_cluster = 0;

        // Find nearest centroid
        for (int k = 0; k < num_clusters; k++)
        {
            double distance = calculate_distance(points[i].coords, centroids[k].coords);
            if (distance < min_distance)
            {
                min_distance = distance;
                nearest_cluster = k;
            }
        }

        // Update cluster assignment
        if (points[i].cluster_id != nearest_cluster)
        {
            points[i].cluster_id = nearest_cluster;
            changes++;
        }
    }

    return changes;
}

// Update centroid positions based on cluster assignments
void update_centroids()
{
    // Reset centroids
    for (int k = 0; k < num_clusters; k++)
    {
        for (int d = 0; d < dimensions; d++)
        {
            centroids[k].coords[d] = 0.0;
        }
        centroids[k].count = 0;
    }

    // Sum all points in each cluster
    for (int i = 0; i < num_points; i++)
    {
        int cluster = points[i].cluster_id;
        if (cluster >= 0)
        {
            for (int d = 0; d < dimensions; d++)
            {
                centroids[cluster].coords[d] += points[i].coords[d];
            }
            centroids[cluster].count++;
        }
    }

    // Calculate mean position
    for (int k = 0; k < num_clusters; k++)
    {
        if (centroids[k].count > 0)
        {
            for (int d = 0; d < dimensions; d++)
            {
                centroids[k].coords[d] /= centroids[k].count;
            }
        }
    }
}

// Calculate Within-Cluster Sum of Squares (WCSS)
double calculate_wcss()
{
    double wcss = 0.0;

    for (int i = 0; i < num_points; i++)
    {
        int cluster = points[i].cluster_id;
        if (cluster >= 0)
        {
            double dist = calculate_distance(points[i].coords, centroids[cluster].coords);
            wcss += dist * dist;
        }
    }

    return wcss;
}

// Write results to file
void write_results(const char *output_file)
{
    FILE *fp = fopen(output_file, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot create output file %s\n", output_file);
        return;
    }

    fprintf(fp, "=== K-Means Clustering Results (Serial) ===\n\n");

    // Write centroids
    fprintf(fp, "Final Centroids:\n");
    for (int k = 0; k < num_clusters; k++)
    {
        fprintf(fp, "Cluster %d (size=%d): ", k, centroids[k].count);
        for (int d = 0; d < dimensions; d++)
        {
            fprintf(fp, "%.6f ", centroids[k].coords[d]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\nWCSS: %.6f\n", calculate_wcss());

    // Write cluster assignments (first 100 points)
    fprintf(fp, "\nSample Point Assignments (first 100):\n");
    int sample_size = (num_points < 100) ? num_points : 100;
    for (int i = 0; i < sample_size; i++)
    {
        fprintf(fp, "Point %d -> Cluster %d: ", i, points[i].cluster_id);
        for (int d = 0; d < dimensions; d++)
        {
            fprintf(fp, "%.6f ", points[i].coords[d]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Results written to: %s\n", output_file);
}

// Cleanup memory
void cleanup()
{
    if (points)
    {
        for (int i = 0; i < num_points; i++)
        {
            free(points[i].coords);
        }
        free(points);
    }

    if (centroids)
    {
        for (int k = 0; k < num_clusters; k++)
        {
            free(centroids[k].coords);
        }
        free(centroids);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <input_file> <num_clusters> <max_iterations>\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s ../datasets/data_10k.txt 5 100\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    num_clusters = atoi(argv[2]);
    int max_iterations = atoi(argv[3]);

    // Validation
    if (num_clusters <= 0)
    {
        fprintf(stderr, "Error: Number of clusters must be positive\n");
        return 1;
    }

    if (max_iterations <= 0)
    {
        fprintf(stderr, "Error: Max iterations must be positive\n");
        return 1;
    }

    printf("=== Serial K-Means Clustering ===\n\n");
    printf("Configuration:\n");
    printf("  Input file: %s\n", input_file);
    printf("  Number of clusters (K): %d\n", num_clusters);
    printf("  Max iterations: %d\n\n", max_iterations);

    // Seed random number generator
    srand(42); // Fixed seed for reproducibility

    // Start timing
    clock_t start_time = clock();

    // Load data
    read_data(input_file);

    if (num_points < num_clusters)
    {
        fprintf(stderr, "Error: Number of clusters cannot exceed number of points\n");
        cleanup();
        return 1;
    }

    // Initialize centroids
    initialize_centroids();

    // K-Means iterations
    printf("\nStarting K-Means iterations...\n");
    int iteration = 0;
    int changes = num_points; // Start with all points changed

    while (iteration < max_iterations && changes > 0)
    {
        // Assign points to clusters
        changes = assign_clusters();

        // Update centroids
        update_centroids();

        // Calculate WCSS
        double wcss = calculate_wcss();

        iteration++;

        if (iteration % 10 == 0 || changes == 0)
        {
            printf("Iteration %d: WCSS = %.6f, Changes = %d\n",
                   iteration, wcss, changes);
        }
    }

    // End timing
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("Converged after %d iterations\n", iteration);
    printf("Final WCSS: %.6f\n", calculate_wcss());
    printf("Execution time: %.6f seconds\n", elapsed_time);

    // Display cluster sizes
    printf("\nCluster Sizes:\n");
    for (int k = 0; k < num_clusters; k++)
    {
        printf("  Cluster %d: %d points\n", k, centroids[k].count);
    }

    // Write results to file
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "../results/serial_results.txt");
    write_results(output_file);

    // Save timing data
    FILE *timing_fp = fopen("../results/serial_timing.txt", "a");
    if (timing_fp)
    {
        fprintf(timing_fp, "%d,%d,%d,%d,%.6f\n",
                num_points, dimensions, num_clusters, iteration, elapsed_time);
        fclose(timing_fp);
    }

    // Cleanup
    cleanup();

    printf("\n=== Done ===\n");
    return 0;
}
