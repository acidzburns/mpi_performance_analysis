#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

double* read_csv(const char* filename, long long* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open %s\n", filename);
        return NULL;
    }
    
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
    
    long long count = 0;
    char* ptr = content;
    while (*ptr) {
        if (*ptr == ',' || *ptr == '\n') count++;
        ptr++;
    }
    if (*(ptr-1) != ',' && *(ptr-1) != '\n') count++;
    
    *size = count;
    
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

void parallel_compute(double* data, long long size) {
    for (long long i = 0; i < size; i++) {
        double val = data[i];
        val = val * val + val * 3.0 - 7.0;
        val = sqrt(fabs(val));
        val = val * 2.0 + 5.0;
        val = val / 2.0;
        if (val > 1000.0) {
            val = fmod(val, 997.0);
        }
        val = val * (i % 5 + 1) + sin(val) * 0.5;
        val = log(fabs(val) + 1.0) * 2.0;
        data[i] = val;
    }
}

void compute_statistics_mpi(double* data, long long size, int rank, int numprocs) {
    double local_sum = 0, local_sum_sq = 0;
    double local_min = data[0], local_max = data[0];
    
    for (long long i = 0; i < size; i++) {
        local_sum += data[i];
        local_sum_sq += data[i] * data[i];
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
    }
    
    double global_sum, global_sum_sq, global_min, global_max;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    long long total_size;
    MPI_Reduce(&size, &total_size, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double mean = global_sum / total_size;
        double variance = (global_sum_sq / total_size) - (mean * mean);
        double stddev = sqrt(variance);
        
        printf("\n  Statistics:\n");
        printf("    Mean: %.4f\n", mean);
        printf("    Variance: %.4f\n", variance);
        printf("    Std Dev: %.4f\n", stddev);
        printf("    Min: %.4f\n", global_min);
        printf("    Max: %.4f\n", global_max);
    }
}

void generate_histogram_mpi(double* data, long long size, int bins, int rank, int numprocs) {
    double local_min = data[0], local_max = data[0];
    for (long long i = 1; i < size; i++) {
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
    }
    
    double global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&global_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double range = global_max - global_min;
    double bin_width = range / bins;
    
    int* local_hist = (int*)calloc(bins, sizeof(int));
    if (!local_hist) return;
    
    for (long long i = 0; i < size; i++) {
        int bin = (int)((data[i] - global_min) / bin_width);
        if (bin >= bins) bin = bins - 1;
        if (bin < 0) bin = 0;
        local_hist[bin]++;
    }
    
    int* global_hist = NULL;
    if (rank == 0) {
        global_hist = (int*)calloc(bins, sizeof(int));
    }
    
    MPI_Reduce(local_hist, global_hist, bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n  Histogram (%d bins):\n", bins);
        for (int i = 0; i < bins; i++) {
            double bin_start = global_min + i * bin_width;
            double bin_end = bin_start + bin_width;
            printf("    [%.2f - %.2f): %d elements\n", bin_start, bin_end, global_hist[i]);
        }
        free(global_hist);
    }
    
    free(local_hist);
}

int compare_double(const void* a, const void* b) {
    double diff = *(double*)a - *(double*)b;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

void parallel_sort_mpi(double* data, long long size, int rank, int numprocs) {
    qsort(data, size, sizeof(double), compare_double);
    
    int* recv_counts = NULL;
    int* displs = NULL;
    double* gathered = NULL;
    
    if (rank == 0) {
        recv_counts = (int*)malloc(numprocs * sizeof(int));
        displs = (int*)malloc(numprocs * sizeof(int));
    }
    
    // Convert long long to int for MPI
    int size_int = (int)size;
    MPI_Gather(&size_int, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < numprocs; i++) {
            displs[i] = total;
            total += recv_counts[i];
        }
        gathered = (double*)malloc(total * sizeof(double));
    }
    
    MPI_Gatherv(data, size_int, MPI_DOUBLE, gathered, recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < numprocs; i++) {
            total += recv_counts[i];
        }
        qsort(gathered, total, sizeof(double), compare_double);
        printf("\n  Parallel Sort: Sorted %d elements\n", total);
        printf("    First 5 sorted values: ");
        for (int i = 0; i < 5 && i < total; i++) {
            printf("%.2f ", gathered[i]);
        }
        printf("\n    Last 5 sorted values: ");
        for (int i = total - 5; i < total && i >= 0; i++) {
            printf("%.2f ", gathered[i]);
        }
        printf("\n");
        
        free(gathered);
        free(recv_counts);
        free(displs);
    }
}

double compute_correlation_mpi(double* data1, double* data2, long long size, int rank, int numprocs) {
    double local_sum_x = 0, local_sum_y = 0, local_sum_xy = 0;
    double local_sum_x2 = 0, local_sum_y2 = 0;
    
    for (long long i = 0; i < size; i++) {
        local_sum_x += data1[i];
        local_sum_y += data2[i];
        local_sum_xy += data1[i] * data2[i];
        local_sum_x2 += data1[i] * data1[i];
        local_sum_y2 += data2[i] * data2[i];
    }
    
    double global_sum_x, global_sum_y, global_sum_xy, global_sum_x2, global_sum_y2;
    long long global_size;
    
    MPI_Reduce(&local_sum_x, &global_sum_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_y, &global_sum_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_xy, &global_sum_xy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_x2, &global_sum_x2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_y2, &global_sum_y2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&size, &global_size, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double correlation = 0;
    if (rank == 0) {
        double n = global_size;
        double numerator = n * global_sum_xy - global_sum_x * global_sum_y;
        double denominator = sqrt((n * global_sum_x2 - global_sum_x * global_sum_x) * 
                                  (n * global_sum_y2 - global_sum_y * global_sum_y));
        if (denominator != 0) {
            correlation = numerator / denominator;
        }
        
        printf("\n  Pearson Correlation: %.4f\n", correlation);
    }
    
    return correlation;
}

void moving_average_mpi(double* data, long long size, int window, int rank, int numprocs) {
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
    
    long long result_size_ll = size - window + 1;
    int result_size = (int)result_size_ll;
    
    int* recv_counts = NULL;
    int* displs = NULL;
    double* gathered = NULL;
    
    if (rank == 0) {
        recv_counts = (int*)malloc(numprocs * sizeof(int));
        displs = (int*)malloc(numprocs * sizeof(int));
    }
    
    MPI_Gather(&result_size, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < numprocs; i++) {
            displs[i] = total;
            total += recv_counts[i];
        }
        gathered = (double*)malloc(total * sizeof(double));
    }
    
    MPI_Gatherv(result, result_size, MPI_DOUBLE, gathered, recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < numprocs; i++) {
            total += recv_counts[i];
        }
        printf("\n  Moving Average (window=%d): Computed %d values\n", window, total);
        printf("    First 5 averages: ");
        for (int i = 0; i < 5 && i < total; i++) {
            printf("%.2f ", gathered[i]);
        }
        printf("\n    Last 5 averages: ");
        int start = total - 5;
        if (start < 0) start = 0;
        for (int i = start; i < total && i >= 0; i++) {
            printf("%.2f ", gathered[i]);
        }
        printf("\n");
        
        free(gathered);
        free(recv_counts);
        free(displs);
    }
    
    free(result);
}

void outlier_detection_mpi(double* data, long long size, int rank, int numprocs) {
    double local_sum = 0, local_sum_sq = 0;
    for (long long i = 0; i < size; i++) {
        local_sum += data[i];
        local_sum_sq += data[i] * data[i];
    }
    
    double global_sum, global_sum_sq;
    long long global_size;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&size, &global_size, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double mean = 0, stddev = 0;
    if (rank == 0) {
        mean = global_sum / global_size;
        double variance = (global_sum_sq / global_size) - (mean * mean);
        stddev = sqrt(variance);
    }
    
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    long long local_outliers = 0;
    for (long long i = 0; i < size; i++) {
        double z_score = (data[i] - mean) / stddev;
        if (fabs(z_score) > 3) {
            local_outliers++;
        }
    }
    
    long long global_outliers;
    MPI_Reduce(&local_outliers, &global_outliers, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n  Outlier Detection (Z-score > 3):\n");
        printf("    Total outliers: %lld (%.2f%% of data)\n", 
               global_outliers, (double)global_outliers / global_size * 100);
    }
}

int main(int argc, char** argv) {
    int rank, numprocs;
    double mpi_start, mpi_end, comp_start, comp_end;
    double transfer_start, transfer_end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    if (rank == 0) {
        printf("\n========================================\n");
        printf("  MPI ANALYTICS - %d Virtual Nodes\n", numprocs);
        printf("========================================\n\n");
    }
    
    char filename[100];
    strcpy(filename, "dataset_10M.csv");
    
    if (argc > 1) {
        sprintf(filename, "dataset_%sM.csv", argv[1]);
    }
    
    long long data_size = 0;
    double* full_data = NULL;
    double* local_data = NULL;
    long long local_size = 0;
    
    transfer_start = MPI_Wtime();
    
    if (rank == 0) {
        printf("Loading dataset: %s\n", filename);
        full_data = read_csv(filename, &data_size);
        if (!full_data) {
            printf("Error: Failed to load dataset\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Loaded %lld elements\n\n", data_size);
    }
    
    MPI_Bcast(&data_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    
    long long chunk_size = data_size / numprocs;
    long long remainder = data_size % numprocs;
    
    long long start_idx = 0;
    if (rank == 0) {
        start_idx = 0;
        local_size = chunk_size + (remainder > 0 ? 1 : 0);
    } else if (rank < remainder) {
        start_idx = rank * (chunk_size + 1);
        local_size = chunk_size + 1;
    } else {
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
        local_size = chunk_size;
    }
    
    local_data = (double*)malloc(local_size * sizeof(double));
    if (!local_data) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        long long current_offset = 0;
        for (int i = 0; i < numprocs; i++) {
            long long count;
            if (i == 0) {
                count = chunk_size + (remainder > 0 ? 1 : 0);
            } else if (i < remainder) {
                count = chunk_size + 1;
            } else {
                count = chunk_size;
            }
            
            if (i == 0) {
                memcpy(local_data, &full_data[current_offset], count * sizeof(double));
            } else {
                MPI_Send(&full_data[current_offset], (int)count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            current_offset += count;
        }
    } else {
        MPI_Recv(local_data, (int)local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    transfer_end = MPI_Wtime();
    double transfer_time = transfer_end - transfer_start;
    
    MPI_Barrier(MPI_COMM_WORLD);
    comp_start = MPI_Wtime();
    
    if (rank == 0) printf("\nTASK 1: Basic Statistics\n");
    if (rank == 0) printf("------------------------\n");
    compute_statistics_mpi(local_data, local_size, rank, numprocs);
    
    if (rank == 0) printf("\nTASK 2: Histogram Generation\n");
    if (rank == 0) printf("----------------------------\n");
    generate_histogram_mpi(local_data, local_size, 10, rank, numprocs);
    
    if (rank == 0) printf("\nTASK 3: Sorting\n");
    if (rank == 0) printf("---------------\n");
    parallel_sort_mpi(local_data, local_size, rank, numprocs);
    
    if (rank == 0) printf("\nTASK 4: Pearson Correlation\n");
    if (rank == 0) printf("----------------------------\n");
    compute_correlation_mpi(local_data, local_data, local_size, rank, numprocs);
    
    if (rank == 0) printf("\nTASK 5: Moving Average\n");
    if (rank == 0) printf("----------------------\n");
    moving_average_mpi(local_data, local_size, 100, rank, numprocs);
    
    if (rank == 0) printf("\nTASK 6: Outlier Detection\n");
    if (rank == 0) printf("-------------------------\n");
    outlier_detection_mpi(local_data, local_size, rank, numprocs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    comp_end = MPI_Wtime();
    double comp_time = comp_end - comp_start;
    double mpi_time = comp_time + transfer_time;
    
    double max_time;
    MPI_Reduce(&mpi_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n========================================\n");
        printf("  MPI EXECUTION COMPLETE\n");
        printf("========================================\n");
        printf("Total MPI Execution Time: %.6f seconds\n", max_time);
        printf("  - Data Transfer: %.6f seconds\n", transfer_time);
        printf("  - Computation: %.6f seconds\n", comp_time);
        printf("========================================\n\n");
        
        free(full_data);
    }
    
    free(local_data);
    MPI_Finalize();
    return 0;
}