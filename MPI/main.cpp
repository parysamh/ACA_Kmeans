#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <limits>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <random>
#include <algorithm>

using std::cout; using std::cin; using std::endl;
using std::vector;

namespace fs = std::filesystem;

static constexpr int DIM = 2;

// squared Euclidean distance between point i (in flat array P) and centroid c (in flat array C)
inline double sqdist2(const double* P, const double* C) {
    double dx = P[0] - C[0];
    double dy = P[1] - C[1];
    return dx*dx + dy*dy;
}

// k-means++ initialization on rank 0
static std::vector<double> random_init_centroids(const std::vector<double>& all_points, int N, int K) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // build [0..N-1], shuffle, take first K indices
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), gen);

    std::vector<double> centroids(K * DIM);
    for (int c = 0; c < K; ++c) {
        int i = idx[c];
        centroids[c*DIM + 0] = all_points[i*DIM + 0];
        centroids[c*DIM + 1] = all_points[i*DIM + 1];
    }
    return centroids;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Fixed defaults
    const int    max_iter = 100;
    const double tol      = 1e-4;

    int N = 0, K = 0;

    // Rank 0 reads only N and K
    if (rank == 0) {
        cout << "Enter number of points: ";
        cin >> N;
        cout << "Enter number of clusters (K): ";
        cin >> K;
        if (K <= 0 || N <= 0 || K > N) {
            std::cerr << "Invalid N/K (require N>0, K>0, K<=N)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast config to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // max_iter and tol are constants; no need to broadcast

    // Distribute counts (how many points per rank)
    vector<int> counts(size), displs(size);
    int base = N / size, rem = N % size, offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    const int local_n = counts[rank];

    // Rank 0 generates points
    vector<double> all_points; all_points.reserve(N * DIM);
    if (rank == 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 100.0);
        all_points.resize(N * DIM);
        for (int i = 0; i < N; ++i) {
            all_points[i*DIM + 0] = dis(gen);
            all_points[i*DIM + 1] = dis(gen);
        }
    }

    // Build Scatterv arrays for doubles (points)
    vector<int> sc_counts_d(size), sc_displs_d(size);
    for (int r = 0; r < size; ++r) {
        sc_counts_d[r] = counts[r] * DIM;
        sc_displs_d[r] = displs[r] * DIM;
    }

    // Local points buffer
    vector<double> local_points(local_n * DIM);

    // Scatter points to ranks
    MPI_Scatterv(rank == 0 ? all_points.data() : nullptr, sc_counts_d.data(), sc_displs_d.data(),
                 MPI_DOUBLE, local_points.data(), local_n * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize centroids on rank 0 (k-means++) and broadcast
    vector<double> centroids(K * DIM);
    if (rank == 0) {
        centroids = random_init_centroids(all_points, N, K);
    }
    MPI_Bcast(centroids.data(), K * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Buffers for iteration
    vector<int>    local_labels(local_n, -1);
    vector<double> local_sum(K * DIM, 0.0), global_sum(K * DIM, 0.0);
    vector<int>    local_cnt(K, 0), global_cnt(K, 0);


    double start_time = MPI_Wtime();

    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        std::fill(local_sum.begin(), local_sum.end(), 0.0);
        std::fill(local_cnt.begin(), local_cnt.end(), 0);

        // Assign labels locally + accumulate partial sums
        for (int i = 0; i < local_n; ++i) {
            double best = std::numeric_limits<double>::infinity();
            int best_c = -1;
            for (int c = 0; c < K; ++c) {
                double d2 = sqdist2(&local_points[i*DIM], &centroids[c*DIM]);
                if (d2 < best) { best = d2; best_c = c; }
            }
            local_labels[i] = best_c;
            local_sum[best_c*DIM + 0] += local_points[i*DIM + 0];
            local_sum[best_c*DIM + 1] += local_points[i*DIM + 1];
            local_cnt[best_c] += 1;
        }

        // Reduce to global
        MPI_Allreduce(local_sum.data(),  global_sum.data(),  K*DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt.data(),  global_cnt.data(),  K,     MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        // Update centroids and compute max shift on rank 0
        double max_shift = 0.0;
        if (rank == 0) {
            for (int c = 0; c < K; ++c) {
                double oldx = centroids[c*DIM+0];
                double oldy = centroids[c*DIM+1];

                if (global_cnt[c] > 0) {
                    centroids[c*DIM+0] = global_sum[c*DIM+0] / global_cnt[c];
                    centroids[c*DIM+1] = global_sum[c*DIM+1] / global_cnt[c];
                }

                double dx = centroids[c*DIM+0] - oldx;
                double dy = centroids[c*DIM+1] - oldy;
                double shift = std::sqrt(dx*dx + dy*dy);
                if (shift > max_shift) max_shift = shift;
            }
        }

        // Broadcast updated centroids and max_shift
        MPI_Bcast(centroids.data(), K * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_shift, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (max_shift < tol) break;
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (rank == 0) {
        cout << "Converged after " << (it + 1) << " iteration(s)\n";
        cout << "Execution time: " << elapsed << " seconds\n";
        cout << "Number of processes (cores) used: " << size << "\n";
    }

    // Gather labels to rank 0
    vector<int> displs_lbl(size), counts_lbl(size);
    for (int r = 0; r < size; ++r) {
        counts_lbl[r] = counts[r];
        displs_lbl[r] = displs[r];
    }
    vector<int> all_labels;
    if (rank == 0) all_labels.resize(N);
    MPI_Gatherv(local_labels.data(), local_n, MPI_INT,
                rank == 0 ? all_labels.data() : nullptr, counts_lbl.data(), displs_lbl.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 writes output next to THIS source file (MPI folder)
    if (rank == 0) {
        fs::path out_path = fs::path(__FILE__).parent_path() / "clustering_result.txt";
        std::ofstream outfile(out_path);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file for writing: " << out_path << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outfile << std::fixed << std::setprecision(1);
        for (int c = 0; c < K; ++c) {
            outfile << "Cluster " << c << " : ("
                    << centroids[c*DIM+0] << ", " << centroids[c*DIM+1] << ") ---> ";

            bool first = true;
            for (int i = 0; i < N; ++i) {
                if (all_labels[i] == c) {
                    if (!first) outfile << " , ";
                    outfile << "Point (" << all_points[i*DIM+0] << ", " << all_points[i*DIM+1] << ")";
                    first = false;
                }
            }
            outfile << "\n";
        }
        outfile.close();
        cout << "Clustering result written to: " << out_path << "\n";
    }

    MPI_Finalize();
    return 0;
}
