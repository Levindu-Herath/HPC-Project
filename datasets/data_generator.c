/*
 * Dataset Generator for K-Means Clustering
 * Generates synthetic data with well-separated clusters
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_DIMENSIONS 10

typedef struct
{
    double center[MAX_DIMENSIONS];
    double std_dev;
} Cluster;

// Generate random number from normal distribution (Box-Muller transform)
double random_normal(double mean, double std_dev)
{
    static int has_spare = 0;
    static double spare;

    if (has_spare)
    {
        has_spare = 0;
        return mean + std_dev * spare;
    }

    has_spare = 1;
    double u, v, s;
    do
    {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std_dev * u * s;
}

void generate_dataset(int num_points, int num_clusters, int dimensions,
                      const char *output_file)
{
    FILE *fp = fopen(output_file, "w");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot create file %s\n", output_file);
        exit(1);
    }

    // Write header
    fprintf(fp, "%d %d\n", num_points, dimensions);

    // Define cluster centers
    Cluster *clusters = (Cluster *)malloc(num_clusters * sizeof(Cluster));

    // Generate well-separated cluster centers
    printf("Generating %d clusters in %dD space...\n", num_clusters, dimensions);
    for (int i = 0; i < num_clusters; i++)
    {
        for (int d = 0; d < dimensions; d++)
        {
            // Spread clusters in a grid-like pattern
            if (dimensions == 2)
            {
                int grid_size = (int)ceil(sqrt(num_clusters));
                clusters[i].center[0] = (i % grid_size) * 20.0 + rand() % 5;
                clusters[i].center[1] = (i / grid_size) * 20.0 + rand() % 5;
            }
            else
            {
                clusters[i].center[d] = (i * 30.0 / num_clusters) +
                                        (rand() % 10 - 5);
            }
        }
        clusters[i].std_dev = 2.0 + (rand() % 10) / 10.0; // Std dev between 2.0 and 3.0

        printf("Cluster %d center: ", i);
        for (int d = 0; d < dimensions; d++)
        {
            printf("%.2f ", clusters[i].center[d]);
        }
        printf("(std_dev: %.2f)\n", clusters[i].std_dev);
    }

    // Generate points
    printf("\nGenerating %d data points...\n", num_points);
    int points_per_cluster = num_points / num_clusters;
    int remaining_points = num_points % num_clusters;

    for (int c = 0; c < num_clusters; c++)
    {
        int points_in_this_cluster = points_per_cluster;
        if (c < remaining_points)
        {
            points_in_this_cluster++;
        }

        for (int p = 0; p < points_in_this_cluster; p++)
        {
            for (int d = 0; d < dimensions; d++)
            {
                double value = random_normal(clusters[c].center[d],
                                             clusters[c].std_dev);
                fprintf(fp, "%.6f", value);
                if (d < dimensions - 1)
                {
                    fprintf(fp, " ");
                }
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    free(clusters);

    printf("\nDataset generated successfully: %s\n", output_file);
    printf("Total points: %d\n", num_points);
    printf("Dimensions: %d\n", dimensions);
    printf("True clusters: %d\n", num_clusters);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Usage: %s <num_points> <num_clusters> <dimensions> <output_file>\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s 10000 5 2 data_10k.txt      # 10K points, 5 clusters, 2D\n", argv[0]);
        printf("  %s 100000 8 3 data_100k.txt    # 100K points, 8 clusters, 3D\n", argv[0]);
        printf("  %s 1000000 10 2 data_1m.txt    # 1M points, 10 clusters, 2D\n", argv[0]);
        return 1;
    }

    int num_points = atoi(argv[1]);
    int num_clusters = atoi(argv[2]);
    int dimensions = atoi(argv[3]);
    const char *output_file = argv[4];

    // Validation
    if (num_points <= 0 || num_points > 100000000)
    {
        fprintf(stderr, "Error: Number of points must be between 1 and 100M\n");
        return 1;
    }

    if (num_clusters <= 0 || num_clusters > 100)
    {
        fprintf(stderr, "Error: Number of clusters must be between 1 and 100\n");
        return 1;
    }

    if (dimensions <= 0 || dimensions > MAX_DIMENSIONS)
    {
        fprintf(stderr, "Error: Dimensions must be between 1 and %d\n", MAX_DIMENSIONS);
        return 1;
    }

    if (num_points < num_clusters)
    {
        fprintf(stderr, "Error: Number of points must be >= number of clusters\n");
        return 1;
    }

    // Seed random number generator
    srand(time(NULL));

    printf("=== K-Means Dataset Generator ===\n\n");
    generate_dataset(num_points, num_clusters, dimensions, output_file);

    return 0;
}
