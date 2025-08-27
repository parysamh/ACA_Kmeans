#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <limits>
#include <algorithm>

using namespace std;

static constexpr int DIM = 2;

// ---------------- Utilities ----------------
inline float sqdist2(const float* a, const float* b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    return dx*dx + dy*dy;
}
inline float l2(const float* a, const float* b) {
    return std::sqrt(sqdist2(a, b));
}

// ---------------- Lloyd assignment ----------------
static void assign_lloyd(
    const vector<vector<float>>& points,
    const vector<vector<float>>& C,
    int K,
    vector<int>& labels,
    vector<vector<float>>& sum,
    vector<int>& cnt
) {
    const int N = (int)points.size();
    for (int i = 0; i < N; ++i) {
        float best = std::numeric_limits<float>::infinity();
        int bestc = 0;
        for (int c = 0; c < K; ++c) {
            float d2 = sqdist2(points[i].data(), C[c].data());
            if (d2 < best) { best = d2; bestc = c; }
        }
        labels[i] = bestc;
        sum[bestc][0] += points[i][0];
        sum[bestc][1] += points[i][1];
        cnt[bestc] += 1;
    }
}

// ---------------- Center helpers ----------------
static void compute_center_separation(
    const vector<vector<float>>& C, int K, vector<float>& s // s[c] = 0.5 * min_{j!=c} d(Cc,Cj)
) {
    s.assign(K, std::numeric_limits<float>::infinity());
    if (K <= 1) { s[0] = std::numeric_limits<float>::infinity(); return; }
    for (int c = 0; c < K; ++c) {
        float best = std::numeric_limits<float>::infinity();
        for (int j = 0; j < K; ++j) if (j != c) {
            float d = l2(C[c].data(), C[j].data());
            if (d < best) best = d;
        }
        s[c] = 0.5f * best;
    }
}
static void compute_center_dists(
    const vector<vector<float>>& C, int K, vector<float>& Dcc // KxK matrix
) {
    Dcc.assign((size_t)K*(size_t)K, 0.0f);
    for (int i = 0; i < K; ++i) {
        for (int j = i+1; j < K; ++j) {
            float d = l2(C[i].data(), C[j].data());
            Dcc[i*(size_t)K + j] = d;
            Dcc[j*(size_t)K + i] = d;
        }
    }
}

// ---------------- Hamerly assignment ----------------
static void assign_hamerly(
    const vector<vector<float>>& P,
    const vector<vector<float>>& C,
    const vector<vector<float>>& prevC,
    const vector<float>& s,            // per-center separation
    const vector<float>& cMove,        // per-center movement ||C - prevC||
    int K,
    vector<int>& labels,
    vector<float>& upper,              // per point
    vector<float>& lower,              // per point
    vector<vector<float>>& sum,
    vector<int>& cnt
) {
    const int N = (int)P.size();
    // lazy update bounds
    float maxMove = 0.0f; for (int c = 0; c < K; ++c) maxMove = std::max(maxMove, cMove[c]);
    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        if (a >= 0) upper[i] += cMove[a];
        lower[i] = std::max(0.0f, lower[i] - maxMove);
    }

    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        bool need = (a < 0);
        if (!need) {
            if (upper[i] <= s[a] && upper[i] <= lower[i]) need = false; else need = true;
        }

        float best = upper[i];
        int bestc = a;
        float second = std::numeric_limits<float>::infinity();

        if (need) {
            if (a >= 0) best = l2(P[i].data(), C[a].data());
            else best = std::numeric_limits<float>::infinity();

            for (int c = 0; c < K; ++c) if (c != a) {
                if (a >= 0) {
                    float ccsep = l2(C[a].data(), C[c].data());
                    if (best <= 0.5f * ccsep && best <= lower[i]) continue;
                }
                float d = l2(P[i].data(), C[c].data());
                if (d < best) { second = best; best = d; bestc = c; }
                else if (d < second) { second = d; }
            }
            labels[i] = bestc;
            upper[i]  = best;
            lower[i]  = second;
        }

        sum[labels[i]][0] += P[i][0];
        sum[labels[i]][1] += P[i][1];
        cnt[labels[i]] += 1;
    }
}

// ---------------- Elkan assignment ----------------
static void assign_elkan(
    const vector<vector<float>>& P,
    const vector<vector<float>>& C,
    const vector<vector<float>>& prevC,
    const vector<float>& cMove,  // per center movement
    const vector<float>& Dcc,    // KxK center distances
    int K,
    vector<int>& labels,
    vector<float>& upper,        // per point
    vector<float>& lowerMat,     // N*K, row-major
    vector<vector<float>>& sum,
    vector<int>& cnt
) {
    const int N = (int)P.size();
    // lazy-update bounds
    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        if (a >= 0) upper[i] += cMove[a];
        float* Li = &lowerMat[(size_t)i * (size_t)K];
        for (int c = 0; c < K; ++c) Li[c] = std::max(0.0f, Li[c] - cMove[c]);
    }

    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        float* Li = &lowerMat[(size_t)i * (size_t)K];

        // First-time: compute all distances
        if (a < 0) {
            float best = std::numeric_limits<float>::infinity();
            int bestc = 0;
            float second = std::numeric_limits<float>::infinity();
            for (int c = 0; c < K; ++c) {
                float d = l2(P[i].data(), C[c].data());
                Li[c] = d;
                if (d < best) { second = best; best = d; bestc = c; }
                else if (d < second) { second = d; }
            }
            labels[i] = a = bestc;
            upper[i] = best;
            sum[a][0] += P[i][0];
            sum[a][1] += P[i][1];
            cnt[a] += 1;
            continue;
        }

        // global test
        const float u0 = upper[i];
        float bound = std::numeric_limits<float>::infinity();
        for (int c = 0; c < K; ++c) if (c != a) {
            float gate = std::max(Li[c], 0.5f * Dcc[(size_t)a*(size_t)K + c]);
            if (gate < bound) bound = gate;
        }
        bool reassess = !(u0 <= bound);

        int bestc = a;
        float best = u0;
        if (reassess) {
            best = l2(P[i].data(), C[a].data()); // tighten
            for (int c = 0; c < K; ++c) if (c != a) {
                float gate = std::max(Li[c], 0.5f * Dcc[(size_t)a*(size_t)K + c]);
                if (best <= gate) continue;
                float d = l2(P[i].data(), C[c].data());
                Li[c] = d;
                if (d < best) {
                    Li[a] = best;  // old best becomes lower bound for old a
                    best = d; bestc = c; a = c;
                }
            }
            upper[i] = best; labels[i] = bestc; Li[bestc] = best;
        }

        sum[labels[i]][0] += P[i][0];
        sum[labels[i]][1] += P[i][1];
        cnt[labels[i]] += 1;
    }
}

// ---------------- Yinyang helpers & assignment ----------------
static void group_centroids_kmeans(
    const vector<vector<float>>& C, int K, int G, vector<int>& gid
) {
    G = std::max(1, std::min(G, K));
    gid.assign(K, 0);
    if (G == 1) { std::fill(gid.begin(), gid.end(), 0); return; }

    vector<vector<float>> GC(G, vector<float>(DIM, 0.0f));
    // init: pick evenly spaced centers
    for (int g = 0; g < G; ++g) {
        int idx = (int)std::floor((double)g * K / G);
        GC[g][0] = C[idx][0];
        GC[g][1] = C[idx][1];
    }
    const int iters = 5;
    vector<int> gcnt(G, 0);
    for (int it = 0; it < iters; ++it) {
        // assign centers to groups
        for (int c = 0; c < K; ++c) {
            float best = std::numeric_limits<float>::infinity(); int bestg = 0;
            for (int g = 0; g < G; ++g) {
                float d2 = sqdist2(C[c].data(), GC[g].data());
                if (d2 < best) { best = d2; bestg = g; }
            }
            gid[c] = bestg;
        }
        // recompute group centroids
        std::fill(GC.begin(), GC.end(), vector<float>(DIM, 0.0f));
        std::fill(gcnt.begin(), gcnt.end(), 0);
        for (int c = 0; c < K; ++c) {
            int g = gid[c];
            GC[g][0] += C[c][0];
            GC[g][1] += C[c][1];
            gcnt[g] += 1;
        }
        for (int g = 0; g < G; ++g) if (gcnt[g] > 0) {
            GC[g][0] /= gcnt[g];
            GC[g][1] /= gcnt[g];
        }
    }
}

static void assign_yinyang(
    const vector<vector<float>>& P,
    const vector<vector<float>>& C,
    const vector<vector<float>>& prevC,
    const vector<int>& gid,   // size K
    int G,
    const vector<float>& cMove,
    int K,
    vector<int>& labels,
    vector<float>& upper,     // per point
    vector<float>& lowerG,    // N*G row-major
    vector<vector<float>>& sum,
    vector<int>& cnt
) {
    const int N = (int)P.size();
    // per-group movement (max center move in group)
    vector<float> gMove(G, 0.0f);
    for (int c = 0; c < K; ++c) gMove[gid[c]] = std::max(gMove[gid[c]], cMove[c]);

    // lazy update
    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        if (a >= 0) upper[i] += cMove[a];
        float* LG = &lowerG[(size_t)i * (size_t)G];
        for (int g = 0; g < G; ++g) LG[g] = std::max(0.0f, LG[g] - gMove[g]);
    }

    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        int ga = (a >= 0 ? gid[a] : -1);
        float u = upper[i];
        float* LG = &lowerG[(size_t)i * (size_t)G];

        // global group test
        float minLG = std::numeric_limits<float>::infinity();
        for (int g = 0; g < G; ++g) minLG = std::min(minLG, LG[g]);
        bool need = (a < 0) || !(u <= minLG);

        float best = u;
        int bestc = a;

        if (need) {
            if (a >= 0) best = l2(P[i].data(), C[a].data());
            else best = std::numeric_limits<float>::infinity();

            for (int g = 0; g < G; ++g) if (LG[g] < best) {
                for (int c = 0; c < K; ++c) if (gid[c] == g) {
                    if (c == a) continue;
                    float d = l2(P[i].data(), C[c].data());
                    if (d < best) { best = d; bestc = c; }
                }
            }

            // refresh group lower-bounds by min distance to any center in group
            for (int g = 0; g < G; ++g) {
                float gbest = std::numeric_limits<float>::infinity();
                for (int c = 0; c < K; ++c) if (gid[c] == g) {
                    float d = l2(P[i].data(), C[c].data());
                    if (d < gbest) gbest = d;
                }
                LG[g] = gbest;
            }

            a = bestc; u = best; labels[i] = a; upper[i] = u; ga = gid[a];
        }

        sum[a][0] += P[i][0];
        sum[a][1] += P[i][1];
        cnt[a] += 1;
    }
}


// ---------------- Main ----------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(&cout);

    int choice = 1, N = 0, K = 0;
    cout << "Choose k-means variant:\n";
    cout << " [1] simple k-means (Lloyd)\n";
    cout << " [2] elkan\n";
    cout << " [3] hamerly\n";
    cout << " [4] yinyang\n";
    cout << "Enter choice: ";
    cin >> choice;
    if (choice < 1 || choice > 4) choice = 1;

    cout << "Enter number of points: ";
    cin >> N;
    cout << "Enter number of clusters (K): ";
    cin >> K;
    if (N <= 0 || K <= 0 || K > N) {
        cerr << "Invalid N/K (require N>0, K>0, K<=N)\n";
        return 1;
    }

    // Data
    vector<vector<float>> points(N, vector<float>(DIM));
    vector<vector<float>> C(K, vector<float>(DIM));
    vector<vector<float>> prevC(K, vector<float>(DIM));

    vector<int> labels(N, -1);
    vector<vector<float>> sum(K, vector<float>(DIM, 0.0f));
    vector<int> cnt(K, 0);

    // Generate uniform random points (like your original code)
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int i = 0; i < N; ++i) {
        points[i][0] = dis(gen);
        points[i][1] = dis(gen);
    }

    // Init: pick K random points as centroids
    std::uniform_int_distribution<> idx_dis(0, N - 1);
    for (int i = 0; i < K; ++i) {
        int idx = idx_dis(gen);
        C[i] = points[idx];
    }
    prevC = C;

    // Variant-specific buffers
    vector<float> s;               // Hamerly separation per center
    vector<float> cMove(K, 0.0f);  // per center movement
    vector<float> upper(N, std::numeric_limits<float>::infinity());
    vector<float> lower(N, 0.0f);

    vector<float> Dcc;             // Elkan (& Yinyang extra) KxK
    vector<float> lowerMat;        // Elkan N*K
    if (choice == 2) lowerMat.assign((size_t)N * (size_t)K, 0.0f);

    int G = std::max(1, std::min(K, 10)); // Yinyang groups
    vector<int> gid(K, 0);
    vector<float> lowerG;          // Yinyang N*G
    if (choice == 4) lowerG.assign((size_t)N * (size_t)G, 0.0f);

    // Run
    const double tol = 1e-4;
    const int max_iter = 100;
    int converged_after = max_iter;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iter; ++it) {
        // reset accumulators
        for (int c = 0; c < K; ++c) { sum[c][0] = sum[c][1] = 0.0f; cnt[c] = 0; }

        // precompute movements and helpers
        for (int c = 0; c < K; ++c) {
            float dx = C[c][0] - prevC[c][0];
            float dy = C[c][1] - prevC[c][1];
            cMove[c] = std::sqrt(dx*dx + dy*dy);
        }
        if (choice == 3) compute_center_separation(C, K, s);
        if (choice == 2 || choice == 4) compute_center_dists(C, K, Dcc);
        if (choice == 4) group_centroids_kmeans(C, K, G, gid);

        // assignment
        switch (choice) {
            case 1:
                assign_lloyd(points, C, K, labels, sum, cnt);
                break;
            case 2:
                assign_elkan(points, C, prevC, cMove, Dcc, K, labels, upper, lowerMat, sum, cnt);
                break;
            case 3:
                assign_hamerly(points, C, prevC, s, cMove, K, labels, upper, lower, sum, cnt);
                break;
            case 4:
                assign_yinyang(points, C, prevC, gid, G, cMove, K, labels, upper, lowerG, sum, cnt);
                break;
            default:
                assign_lloyd(points, C, K, labels, sum, cnt);
        }

        // update centers & convergence
        float max_shift = 0.0f;
        for (int c = 0; c < K; ++c) {
            prevC[c] = C[c];
            if (cnt[c] > 0) {
                C[c][0] = sum[c][0] / cnt[c];
                C[c][1] = sum[c][1] / cnt[c];
            }
            float dx = C[c][0] - prevC[c][0];
            float dy = C[c][1] - prevC[c][1];
            float sh = std::sqrt(dx*dx + dy*dy);
            if (sh > max_shift) max_shift = sh;
        }

        if (max_shift < tol) { converged_after = it + 1; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ----- summary (with separator you wanted) -----
    cout << "---------------------------------------------------------------------\n";
    cout << "Converged after " << converged_after << " iteration(s)\n";
    cout << "Variant: " << (choice==1?"Lloyd": choice==2?"Elkan": choice==3?"Hamerly":"Yinyang") << "\n";
    cout << "Execution time (clustering only): " << ms << " ms\n";

    // write file
    ofstream outfile("clustering_result.txt");
    if (!outfile.is_open()) {
        cerr << "Error opening file for writing!\n";
        return 1;
    }
    outfile << fixed << setprecision(1);
    for (int c = 0; c < K; ++c) {
        outfile << "Cluster " << (c) << " : (" << C[c][0] << ", " << C[c][1] << ") ---> ";
        bool first = true;
        for (int i = 0; i < N; ++i) {
            if (labels[i] == c) {
                if (!first) outfile << " , ";
                outfile << "Point (" << points[i][0] << ", " << points[i][1] << ")";
                first = false;
            }
        }
        outfile << "\n";
    }
    outfile.close();
    return 0;
}
