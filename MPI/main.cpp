// MPI K-Means Variants: Lloyd (plain), Elkan, Hamerly, Yinyang
// Single-file C++17 + MPI implementation (2D points for brevity; can generalize DIM)
// ---------------------------------------------------------------
// Build:
//   mpic++ -std=gnu++17 -O3 -march=native -o kmeans_variants kmeans_variants.cpp
// Run:
//   mpirun -np 4 ./kmeans_variants
// ---------------------------------------------------------------
// Notes:
// - This file implements exact variants (no approximation) of k-means:
//     [1] Lloyd (plain)
//     [2] Elkan (full lower-bound matrix per local point)
//     [3] Hamerly (one lower+upper per local point)
//     [4] Yinyang (grouped-centroid pruning with per-group lower bounds)
// - Each rank holds only its local partition of points; centroids are global and broadcast each iteration.
// - For simplicity, points are synthetically generated on rank 0 and Scatterv'ed.
// - Yinyang grouping uses a lightweight k-means on centroids (on rank 0) to build G groups each iteration.
// - All variants are exact and should converge to the same solution as Lloyd for the same initialization.
// - This is an educational implementation optimized for clarity; production code can push more SIMD/OpenMP.

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector; using std::cout; using std::cin; using std::endl;
namespace fs = std::filesystem;

// -------------------------- Config --------------------------
static constexpr int DIM = 2;  // set to 2 for demo and small memory; can generalize

// ---------------------- Math Utilities ----------------------
inline double sqdist2(const double* p, const double* c) {
    double dx = p[0] - c[0];
    double dy = p[1] - c[1];
    return dx*dx + dy*dy;
}

inline double l2(const double* a, const double* b) {
    return std::sqrt(sqdist2(a, b));
}


// ------------------- Center Separation (Elkan/Hamerly) -------------------
static void compute_center_separation(const vector<double>& C, int K, vector<double>& s) {
    s.assign(K, std::numeric_limits<double>::infinity());
    for (int c = 0; c < K; ++c) {
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < K; ++j) if (j != c) {
            double d = l2(&C[c*DIM], &C[j*DIM]);
            if (d < best) best = d;
        }
        s[c] = 0.5 * best;
    }
}

static void compute_center_dists(const vector<double>& C, int K, vector<double>& Dcc) {
    Dcc.assign(K*K, 0.0);
    for (int i = 0; i < K; ++i) {
        for (int j = i+1; j < K; ++j) {
            double d = l2(&C[i*DIM], &C[j*DIM]);
            Dcc[i*K + j] = Dcc[j*K + i] = d;
        }
    }
}

// ---------------------- Lloyd Assignment ----------------------
static void assign_lloyd(
    const vector<double>& local_points,
    const vector<double>& C,
    int K,
    vector<int>& label,
    vector<double>& sum,
    vector<int>& cnt
) {
    const int n = (int)label.size();
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0);
    for (int i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int bestc = 0;
        for (int c = 0; c < K; ++c) {
            double d2 = sqdist2(&local_points[i*DIM], &C[c*DIM]);
            if (d2 < best) { best = d2; bestc = c; }
        }
        label[i] = bestc;
        sum[bestc*DIM+0] += local_points[i*DIM+0];
        sum[bestc*DIM+1] += local_points[i*DIM+1];
        cnt[bestc]++;
    }
}

// ---------------------- Hamerly Assignment ----------------------
static void assign_hamerly(
    const vector<double>& P,
    const vector<double>& C,
    const vector<double>& prevC,
    const vector<double>& s,              // 0.5 * nearest-center distance per center
    const vector<double>& cMove,          // ||C - prevC|| per center
    int K,
    vector<int>& label,
    vector<double>& upper,
    vector<double>& lower,
    vector<double>& sum,
    vector<int>& cnt
) {
    const int n = (int)label.size();
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0);

    // lazy-update bounds
    double maxMove = 0.0; for (int c = 0; c < K; ++c) maxMove = std::max(maxMove, cMove[c]);
    for (int i = 0; i < n; ++i) {
        int a = label[i];
        if (a >= 0) upper[i] += cMove[a];
        lower[i] = std::max(0.0, lower[i] - maxMove);
    }

    for (int i = 0; i < n; ++i) {
        int a = label[i];
        bool need = (a < 0);
        if (!need) {
            if (upper[i] <= s[a] && upper[i] <= lower[i]) need = false; else need = true;
        }

        double u = upper[i];
        int bestc = a;
        double best = u;
        double second = std::numeric_limits<double>::infinity();

        if (need) {
            if (a >= 0) { best = std::sqrt(sqdist2(&P[i*DIM], &C[a*DIM])); }
            else { best = std::numeric_limits<double>::infinity(); }

            for (int c = 0; c < K; ++c) if (c != a) {
                // prune by triangle inequality: if best <= 0.5 * dist(Ca, Cc) and best <= lower[i], skip
                if (a >= 0) {
                    double ccsep = l2(&C[a*DIM], &C[c*DIM]);
                    if (best <= 0.5 * ccsep && best <= lower[i]) continue;
                }
                double d = std::sqrt(sqdist2(&P[i*DIM], &C[c*DIM]));
                if (d < best) { second = best; best = d; bestc = c; }
                else if (d < second) { second = d; }
            }
            u = best; label[i] = bestc; upper[i] = u; lower[i] = second;
        }

        sum[bestc*DIM+0] += P[i*DIM+0];
        sum[bestc*DIM+1] += P[i*DIM+1];
        cnt[bestc]++;
    }
}

// ---------------------- Elkan Assignment ----------------------
// Exact Elkan: per-point upper bound u[i] and lower bounds L[i][c] for all c, plus center-center distance matrix.
static void assign_elkan(
    const vector<double>& P,
    const vector<double>& C,
    const vector<double>& prevC,
    const vector<double>& cMove,  // ||C - prevC|| per center
    const vector<double>& Dcc,    // center-center distances (KxK)
    int K,
    vector<int>& label,
    vector<double>& upper,
    vector<double>& lowerMat, // size: n*K (row-major per point)
    vector<double>& sum,
    vector<int>& cnt
) {
    const int n = (int)label.size();
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0);

    // Lazy update bounds per center movement
    for (int i = 0; i < n; ++i) {
        int a = label[i];
        if (a >= 0) upper[i] += cMove[a];
        double* Li = &lowerMat[i*(size_t)K];
        for (int c = 0; c < K; ++c) {
            Li[c] = std::max(0.0, Li[c] - cMove[c]);
        }
    }

    for (int i = 0; i < n; ++i) {
        int a = label[i];
        double* Li = &lowerMat[i*(size_t)K];

        // First-time assignment: compute all distances, set best and bounds
        if (a < 0) {
            double best = std::numeric_limits<double>::infinity();
            int bestc = 0; double second = std::numeric_limits<double>::infinity();
            for (int c = 0; c < K; ++c) {
                double d = std::sqrt(sqdist2(&P[i*DIM], &C[c*DIM]));
                Li[c] = d; // lower bound equals exact distance after evaluation
                if (d < best) { second = best; best = d; bestc = c; }
                else if (d < second) { second = d; }
            }
            label[i] = a = bestc;
            upper[i] = best;
            // continue to accumulation
            sum[a*DIM+0] += P[i*DIM+0];
            sum[a*DIM+1] += P[i*DIM+1];
            cnt[a]++;
            continue;
        }

        // Global test: u <= min_{c != a} max(L[i,c], 0.5*Dcc[a,c])
        const double u0 = upper[i];
        double bound = std::numeric_limits<double>::infinity();
        for (int c = 0; c < K; ++c) if (c != a) {
            double gate = std::max(Li[c], 0.5 * Dcc[a*K + c]);
            if (gate < bound) bound = gate;
        }
        bool reassess = !(u0 <= bound);

        int bestc = a;
        double best = u0;
        if (reassess) {
            // Tighten true upper bound to assigned center
            best = std::sqrt(sqdist2(&P[i*DIM], &C[a*DIM]));
            // Try others with pruning
            for (int c = 0; c < K; ++c) if (c != a) {
                double gate = std::max(Li[c], 0.5 * Dcc[a*K + c]);
                if (best <= gate) continue;
                double d = std::sqrt(sqdist2(&P[i*DIM], &C[c*DIM]));
                Li[c] = d; // tighten lower bound for c
                if (d < best) {
                    Li[a] = best; // old best becomes lower bound for old a
                    best = d; bestc = c; a = c; // switch current center for stronger pruning
                }
            }
            upper[i] = best; label[i] = bestc; Li[bestc] = best;
        }

        sum[label[i]*DIM+0] += P[i*DIM+0];
        sum[label[i]*DIM+1] += P[i*DIM+1];
        cnt[label[i]]++;
    }
}

// ---------------------- Yinyang Assignment ----------------------
// Group centroids into G groups; keep per-point upper bound u[i] and per-group lower bounds Lg[i][g].
// Tests:
//   If u[i] <= min_g Lg[i][g]  → no change
//   If u[i] <= Lg[i][g*] for the group of assigned center → no need to inspect other groups
// Implementation outline:
//   - Build groups on rank 0 by running a tiny k-means on centers to G groups; broadcast group ids.
//   - Maintain center movement and group movement (max move within group), update Lg lazily.
//   - Within candidate groups, use Elkan-like per-center pruning relative to current best.

static void kmeans_group_centroids(const vector<double>& C, int K, int G, vector<int>& gid) {
    // Simple grouping of centers via Lloyd on centers themselves
    // Initialize group centers by picking G evenly spaced centers
    G = std::max(1, std::min(G, K));
    gid.assign(K, 0);
    if (G == 1) { std::fill(gid.begin(), gid.end(), 0); return; }

    vector<double> GC(G * DIM, 0.0);
    for (int g = 0; g < G; ++g) {
        int idx = (int)std::floor((double)g * K / G);
        GC[g*DIM+0] = C[idx*DIM+0];
        GC[g*DIM+1] = C[idx*DIM+1];
    }
    const int iters = 5; // few steps are enough
    vector<int> gcnt(G,0);
    for (int it = 0; it < iters; ++it) {
        std::fill(gid.begin(), gid.end(), 0);
        // assign centers to groups
        for (int c = 0; c < K; ++c) {
            double best = std::numeric_limits<double>::infinity(); int bestg = 0;
            for (int g = 0; g < G; ++g) {
                double d2 = sqdist2(&C[c*DIM], &GC[g*DIM]);
                if (d2 < best) { best = d2; bestg = g; }
            }
            gid[c] = bestg;
        }
        // recompute group centers
        std::fill(GC.begin(), GC.end(), 0.0);
        std::fill(gcnt.begin(), gcnt.end(), 0);
        for (int c = 0; c < K; ++c) {
            int g = gid[c];
            GC[g*DIM+0] += C[c*DIM+0];
            GC[g*DIM+1] += C[c*DIM+1];
            gcnt[g]++;
        }
        for (int g = 0; g < G; ++g) if (gcnt[g] > 0) {
            GC[g*DIM+0] /= gcnt[g];
            GC[g*DIM+1] /= gcnt[g];
        }
    }
}

static void assign_yinyang(
    const vector<double>& P,
    const vector<double>& C,
    const vector<double>& prevC,
    const vector<int>& gid,     // size K; group id per center
    int G,                      // number of groups
    const vector<double>& cMove,// per center movement
    const vector<double>& Dcc,  // center-center distances (optional for extra pruning)
    int K,
    vector<int>& label,
    vector<double>& upper,        // per point
    vector<double>& lowerG,       // per point group lower bounds (n*G)
    vector<double>& sum,
    vector<int>& cnt
) {
    const int n = (int)label.size();
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0);

    // Precompute group movements (max center move per group)
    vector<double> gMove(G, 0.0);
    for (int c = 0; c < K; ++c) gMove[gid[c]] = std::max(gMove[gid[c]], cMove[c]);

    // Lazy update per-point bounds
    for (int i = 0; i < n; ++i) {
        int a = label[i];
        if (a >= 0) upper[i] += cMove[a];
        double* LG = &lowerG[i*(size_t)G];
        for (int g = 0; g < G; ++g) LG[g] = std::max(0.0, LG[g] - gMove[g]);
    }

    for (int i = 0; i < n; ++i) {
        int a = label[i];
        int ga = (a >= 0 ? gid[a] : -1);

        // Global group test: if u <= min_g Lg → no change
        double u = upper[i];
        double* LG = &lowerG[i*(size_t)G];
        double minLG = std::numeric_limits<double>::infinity();
        for (int g = 0; g < G; ++g) minLG = std::min(minLG, LG[g]);
        bool need = (a < 0) || !(u <= minLG);

        double best = u; int bestc = a;
        if (need) {
            // Tighten true distance to current center if any
            if (a >= 0) best = std::sqrt(sqdist2(&P[i*DIM], &C[a*DIM]));
            else best = std::numeric_limits<double>::infinity();

            // Candidate groups: those with LG[g] < best
            for (int g = 0; g < G; ++g) if (LG[g] < best) {
                // Inspect centers in group g
                for (int c = 0; c < K; ++c) if (gid[c] == g) {
                    if (c == a) continue;
                    double d = std::sqrt(sqdist2(&P[i*DIM], &C[c*DIM]));
                    if (d < best) { best = d; bestc = c; }
                }
            }

            // After inspecting, re-estimate group lower bounds as best distance to any center in group
            for (int g = 0; g < G; ++g) {
                double gbest = std::numeric_limits<double>::infinity();
                for (int c = 0; c < K; ++c) if (gid[c] == g) {
                    double d = std::sqrt(sqdist2(&P[i*DIM], &C[c*DIM]));
                    if (d < gbest) gbest = d;
                }
                LG[g] = gbest;
            }

            a = bestc; u = best; label[i] = a; upper[i] = u; ga = gid[a];
        }

        // accumulate
        sum[a*DIM+0] += P[i*DIM+0];
        sum[a*DIM+1] += P[i*DIM+1];
        cnt[a]++;
    }
}

// ----------------------------- Main -----------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int max_iter = 100; const double tol = 1e-4;
    int choice = 1; int N = 0; int K = 0;

    if (rank == 0) {
        cout << "Choose k-means variant:\n";
        cout << " [1] simple k-means (Lloyd)\n";
        cout << " [2] elkan\n";
        cout << " [3] hamerly\n";
        cout << " [4] yinyang\n";
        cout << "Enter choice: "; cin >> choice; if (choice < 1 || choice > 4) choice = 1;
        cout << "Enter number of points: "; cin >> N;
        cout << "Enter number of clusters (K): "; cin >> K;
        if (K <= 0 || N <= 0 || K > N) { std::cerr << "Invalid N/K\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition points
    vector<int> counts(size), displs(size);
    int base = N / size, rem = N % size, off = 0;
    for (int r = 0; r < size; ++r) { counts[r] = base + (r < rem ? 1 : 0); displs[r] = off; off += counts[r]; }
    int nloc = counts[rank];

    // Generate on rank 0
    vector<double> allP; allP.reserve((size_t)N*DIM);
    if (rank == 0) {
        std::mt19937 gen(42);
        // Gaussian blobs to make accelerations shine
        int blobs = std::max(1, K);
        std::normal_distribution<double> ga(0.0, 1.0);
        std::uniform_real_distribution<double> center(10.0, 90.0);
        vector<double> blobC(blobs*DIM);
        for (int b=0;b<blobs;++b){ blobC[b*DIM]=center(gen); blobC[b*DIM+1]=center(gen);}
        allP.resize((size_t)N*DIM);
        for (int i=0;i<N;++i){ int b=i%blobs; allP[i*DIM]=blobC[b*DIM]+ga(gen)*3.0; allP[i*DIM+1]=blobC[b*DIM+1]+ga(gen)*3.0; }
    }

    // Scatterv points
    vector<int> sc_counts(size), sc_displs(size);
    for (int r=0;r<size;++r){ sc_counts[r]=counts[r]*DIM; sc_displs[r]=displs[r]*DIM; }
    vector<double> P(nloc*DIM);
    MPI_Scatterv(rank==0?allP.data():nullptr, sc_counts.data(), sc_displs.data(), MPI_DOUBLE,
                 P.data(), nloc*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize centroids on rank 0 (random points, no k-means++)
    vector<double> C(K*DIM), prevC(K*DIM);
    if (rank == 0) {
        std::mt19937 gen(2025);
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), gen);    // sample without replacement
        for (int i = 0; i < K; ++i) {
            int id = idx[i];
            C[i*DIM + 0] = allP[id*DIM + 0];
            C[i*DIM + 1] = allP[id*DIM + 1];
        }
    }
    MPI_Bcast(C.data(), K*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    prevC = C;

    // Common buffers
    vector<int> label(nloc, -1);
    vector<double> sum(K*DIM, 0.0), gsum(K*DIM, 0.0);
    vector<int> cnt(K, 0), gcnt(K, 0);

    // Variant-specific state
    vector<double> upper(nloc, std::numeric_limits<double>::infinity());
    vector<double> lower(nloc, 0.0);                   // Hamerly
    vector<double> cMove(K, 0.0), s(K, 0.0);           // movements & separations

    vector<double> Dcc;                                 // Elkan & YY extra pruning
    vector<double> lowerMat;                            // Elkan: nloc*K

    int G = std::max(1, std::min(K, 10));              // Yinyang groups (configurable)
    vector<int> gid(K, 0);                              // group id per center
    vector<double> lowerG;                              // Yinyang: nloc*G

    // Initialize lower matrices when needed
    if (choice == 2) lowerMat.assign((size_t)nloc*K, 0.0);
    if (choice == 4) lowerG.assign((size_t)nloc*G, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int it = 0; double max_shift = 0.0;
    for (it = 0; it < max_iter; ++it) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);

        // Precompute center movement and helpers on rank 0
        if (rank == 0) {
            for (int c = 0; c < K; ++c) {
                double dx = C[c*DIM+0] - prevC[c*DIM+0];
                double dy = C[c*DIM+1] - prevC[c*DIM+1];
                cMove[c] = std::sqrt(dx*dx + dy*dy);
            }
            if (choice == 3) { compute_center_separation(C, K, s); }
            if (choice == 2 || choice == 4) { compute_center_dists(C, K, Dcc); }
            if (choice == 4) { kmeans_group_centroids(C, K, G, gid); }
        }
        // Broadcast helpers
        // Ensure receive-side buffers are sized BEFORE MPI_Bcast
        if (choice == 3) { if (rank != 0) s.resize(K); }
        if (choice == 2 || choice == 4) { if (rank != 0) Dcc.resize((size_t)K*(size_t)K); }
        if (choice == 4) { if (rank != 0) gid.resize(K); }

        MPI_Bcast(cMove.data(), K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (choice == 3) MPI_Bcast(s.data(), K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (choice == 2 || choice == 4) MPI_Bcast(Dcc.data(), K*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (choice == 4) MPI_Bcast(gid.data(), K, MPI_INT, 0, MPI_COMM_WORLD);

        // Assignment per variant
        switch (choice) {
            case 1:
                assign_lloyd(P, C, K, label, sum, cnt);
                break;
            case 2:
                assign_elkan(P, C, prevC, cMove, Dcc, K, label, upper, lowerMat, sum, cnt);
                break;
            case 3:
                assign_hamerly(P, C, prevC, s, cMove, K, label, upper, lower, sum, cnt);
                break;
            case 4:
                assign_yinyang(P, C, prevC, gid, G, cMove, Dcc, K, label, upper, lowerG, sum, cnt);
                break;
            default:
                assign_lloyd(P, C, K, label, sum, cnt);
        }

        // Allreduce partial sums & counts
        MPI_Allreduce(sum.data(), gsum.data(), K*DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt.data(), gcnt.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update centers on rank 0
        if (rank == 0) {
            prevC = C; max_shift = 0.0;
            for (int c = 0; c < K; ++c) {
                double oldx = C[c*DIM+0], oldy = C[c*DIM+1];
                if (gcnt[c] > 0) {
                    C[c*DIM+0] = gsum[c*DIM+0] / gcnt[c];
                    C[c*DIM+1] = gsum[c*DIM+1] / gcnt[c];
                }
                double dx = C[c*DIM+0] - oldx; double dy = C[c*DIM+1] - oldy;
                double sh = std::sqrt(dx*dx + dy*dy); if (sh > max_shift) max_shift = sh;
            }
        }
        MPI_Bcast(C.data(), K*DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_shift, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (max_shift < tol) break;
    }

    double t1 = MPI_Wtime();
    if (rank == 0) {
		cout << "--------------------------------------------------------------------------\n";
        cout << "Converged after " << (it+1) << " iteration(s)\n";
        cout << "Variant: " << (choice==1?"Lloyd": choice==2?"Elkan": choice==3?"Hamerly":"Yinyang") << "\n";
        cout << "Time (s): " << (t1 - t0) << "\n";
        cout << "MPI ranks: " << size << "\n";
    }

    // Gather labels to rank 0 for output
    vector<int> counts_lbl(size), displs_lbl(size); for (int r=0;r<size;++r){ counts_lbl[r]=counts[r]; displs_lbl[r]=displs[r]; }
    vector<int> allLab; if (rank==0) allLab.resize(N);
    MPI_Gatherv(label.data(), nloc, MPI_INT,
                rank==0?allLab.data():nullptr, counts_lbl.data(), displs_lbl.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fs::path out = fs::current_path() / "clustering_result.txt";
        std::ofstream ofs(out);
        ofs << std::fixed << std::setprecision(2);
        for (int c = 0; c < K; ++c) {
            ofs << "Cluster " << (c+1) << " : (" << C[c*DIM+0] << ", " << C[c*DIM+1] << ") --> ";
            bool first = true;
            for (int i = 0; i < N; ++i) if (allLab[i] == c) {
                if (!first) ofs << ", "; first = false;
                ofs << "(" << allP[i*DIM+0] << ", " << allP[i*DIM+1] << ")";
            }
            ofs << "\n";
        }
        ofs.close();
    }

    MPI_Finalize();
    return 0;
}
