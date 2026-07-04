#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Function prototypes
double* read_csv(const char* filename, long long* size);
void compute_statistics(double* data, long long size);
void generate_histogram(double* data, long long size, int bins);
void parallel_sort(double* data, long long size);
double compute_correlation(double* data1, double* data2, long long size);
void moving_average(double* data, long long size, int window);
void outlier_detection(double* data, long long size);

double* read_csv(const char* filename, long long* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = (char*)malloc(file_size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    fclose(file);
    
    // Count elements
    long long count = 0;
    char* ptr = content;
    while (*ptr) {
        if (*ptr == ',' || *ptr == '\n') count++;
        ptr++;
    }
    if (*(ptr-1) != ',' && *(ptr-1) != '\n') count++;
    
    *size = count;
    
    // Parse data
    double* data = (double*)malloc(count * sizeof(double));
    if (!data) {
        free(content);
        return NULL;
    }
    
    long long idx = 0;
    ptr = content;
    char* token = strtok(content, ",\n");
    while (token != NULL && idx < count) {
        data[idx++] = atof(token);
        token = strtok(NULL, ",\n");
    }
    
    free(content);
    return data;
}

void compute_statistics(double* data, long long size) {
    double sum = 0, sum_sq = 0;
    double min_val = data[0], max_val = data[0];
    
    for (long long i = 0; i < size; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);
    double stddev = sqrt(variance);
    
    printf("\n  Statistics:\n");
    printf("    Mean: %.4f\n", mean);
    printf("    Variance: %.4f\n", variance);
    printf("    Std Dev: %.4f\n", stddev);
    printf("    Min: %.4f\n", min_val);
    printf("    Max: %.4f\n", max_val);
}

void generate_histogram(double* data, long long size, int bins) {
    double min_val = data[0], max_val = data[0];
    for (long long i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    double range = max_val - min_val;
    double bin_width = range / bins;
    
    int* histogram = (int*)calloc(bins, sizeof(int));
    if (!histogram) return;
    
    for (long long i = 0; i < size; i++) {
        int bin = (int)((data[i] - min_val) / bin_width);
        if (bin >= bins) bin = bins - 1;
        histogram[bin]++;
    }
    
    printf("\n  Histogram (%d bins):\n", bins);
    for (int i = 0; i < bins; i++) {
        double bin_start = min_val + i * bin_width;
        double bin_end = bin_start + bin_width;
        printf("    [%.2f - %.2f): %d elements\n", bin_start, bin_end, histogram[i]);
    }
    
    free(histogram);
}

int compare_double(const void* a, const void* b) {
    double diff = *(double*)a - *(double*)b;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

void parallel_sort(double* data, long long size) {
    qsort(data, size, sizeof(double), compare_double);
    printf("\n  Sorting: Completed sort of %lld elements\n", size);
}

double compute_correlation(double* data1, double* data2, long long size) {
    double sum_x = 0, sum_y = 0, sum_xy = 0;
    double sum_x2 = 0, sum_y2 = 0;
    
    for (long long i = 0; i < size; i++) {
        sum_x += data1[i];
        sum_y += data2[i];
        sum_xy += data1[i] * data2[i];
        sum_x2 += data1[i] * data1[i];
        sum_y2 += data2[i] * data2[i];
    }
    
    double n = size;
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    if (denominator == 0) return 0;
    double correlation = numerator / denominator;
    
    printf("\n  Pearson Correlation: %.4f\n", correlation);
    return correlation;
}

void moving_average(double* data, long long size, int window) {
    if (window > size) window = (int)size;
    
    double* result = (double*)malloc((size - window + 1) * sizeof(double));
    if (!result) return;
    
    for (long long i = 0; i < size - window + 1; i++) {
        double sum = 0;
        for (int j = 0; j < window; j++) {
            sum += data[i + j];
        }
        result[i] = sum / window;
    }
    
    printf("\n  Moving Average (window=%d): Computed %lld values\n", 
           window, size - window + 1);
    
    printf("    First 5 averages: ");
    for (int i = 0; i < 5 && i < size - window + 1; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n    Last 5 averages: ");
    long long start = (size - window + 1) - 5;
    if (start < 0) start = 0;
    for (long long i = start; i < size - window + 1 && i >= 0; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");
    
    free(result);
}

void outlier_detection(double* data, long long size) {
    double sum = 0, sum_sq = 0;
    for (long long i = 0; i < size; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }
    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);
    double stddev = sqrt(variance);
    
    long long outlier_count = 0;
    double* outliers = (double*)malloc(size * sizeof(double));
    if (!outliers) return;
    
    for (long long i = 0; i < size; i++) {
        double z_score = (data[i] - mean) / stddev;
        if (fabs(z_score) > 3) {
            outliers[outlier_count++] = data[i];
        }
    }
    
    printf("\n  Outlier Detection (Z-score > 3):\n");
    printf("    Total outliers: %lld (%.2f%% of data)\n", 
           outlier_count, (double)outlier_count / size * 100);
    
    if (outlier_count > 0) {
        printf("    Sample outliers: ");
        for (long long i = 0; i < 5 && i < outlier_count; i++) {
            printf("%.2f ", outliers[i]);
        }
        printf("\n");
    }
    
    free(outliers);
}

int main(int argc, char** argv) {
    printf("\n========================================\n");
    printf("  SEQUENTIAL ANALYTICS - 10M Dataset\n");
    printf("========================================\n\n");
    
    char filename[100];
    strcpy(filename, "dataset_10M.csv");
    
    if (argc > 1) {
        sprintf(filename, "dataset_%sM.csv", argv[1]);
    }
    
    printf("Loading dataset: %s\n", filename);
    
    long long data_size = 0;
    double* data = read_csv(filename, &data_size);
    
    if (!data) {
        printf("Error: Failed to load dataset\n");
        return 1;
    }
    
    printf("Loaded %lld elements\n\n", data_size);
    
    clock_t start_time = clock();
    
    printf("TASK 1: Basic Statistics\n");
    printf("------------------------\n");
    compute_statistics(data, data_size);
    
    printf("\nTASK 2: Histogram Generation\n");
    printf("----------------------------\n");
    generate_histogram(data, data_size, 10);
    
    printf("\nTASK 3: Sorting\n");
    printf("---------------\n");
    parallel_sort(data, data_size);
    
    printf("\nTASK 4: Pearson Correlation\n");
    printf("----------------------------\n");
    compute_correlation(data, data, data_size);
    
    printf("\nTASK 5: Moving Average\n");
    printf("----------------------\n");
    moving_average(data, data_size, 100);
    
    printf("\nTASK 6: Outlier Detection\n");
    printf("-------------------------\n");
    outlier_detection(data, data_size);
    
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\n========================================\n");
    printf("  SEQUENTIAL EXECUTION COMPLETE\n");
    printf("========================================\n");
    printf("Total Execution Time: %.6f seconds\n", elapsed_time);
    printf("========================================\n\n");
    
    free(data);
    return 0;
}