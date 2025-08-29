#include <algorithm>
#include <chrono>
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
#include <unordered_map>
#include <vector>

using clk  = std::chrono::high_resolution_clock;
using secd = std::chrono::duration<double>;

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace fs = std::filesystem;

// string trimming helper for input parsing
static inline void trim_inplace(std::string& s) {
  size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) { s.clear(); return; }
  size_t b = s.find_last_not_of(" \t\r");
  s = s.substr(a, b - a + 1);
}

// global dimension (set from CLI args, e.g. --cols)
static int gDIM = 2;

//math
inline double sqdist2(const double* p, const double* c) {
  double s = 0.0;
  for (int d = 0; d < gDIM; ++d) {
    double dx = p[d] - c[d];
    s += dx * dx;
  }
  return s;
}

inline double l2(const double* a, const double* b) {
  return std::sqrt(sqdist2(a, b));
}

// for Hamerly/Elkan pruning
static void compute_center_separation(const vector<double>& C, int K, vector<double>& s) {
  s.assign(K, std::numeric_limits<double>::infinity());
  for (int c = 0; c < K; ++c) {
    double best = std::numeric_limits<double>::infinity();
    for (int j = 0; j < K; ++j) {
      if (j == c) continue;
      double d = l2(&C[c * (size_t)gDIM], &C[j * (size_t)gDIM]);
      if (d < best) best = d;
    }
    s[c] = 0.5 * best;
  }
}

static void compute_center_dists(const vector<double>& C, int K, vector<double>& Dcc) {
  Dcc.assign((size_t)K * (size_t)K, 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = i + 1; j < K; ++j) {
      double d = l2(&C[i * (size_t)gDIM], &C[j * (size_t)gDIM]);
      Dcc[i * (size_t)K + j] = Dcc[j * (size_t)K + i] = d;
    }
  }
}

//Lloyd
static void assign_lloyd(const vector<double>& P,
                         const vector<double>& C,
                         int K,
                         vector<int>& label,
                         vector<double>& sum,
                         vector<int>& cnt) {
  const int n = (int)label.size();
  std::fill(sum.begin(), sum.end(), 0.0);
  std::fill(cnt.begin(), cnt.end(), 0);

  for (int i = 0; i < n; ++i) {
    double best = std::numeric_limits<double>::infinity();
    int bestc = 0;
    const double* pi = &P[i * (size_t)gDIM];

    for (int c = 0; c < K; ++c) {
      double d2 = sqdist2(pi, &C[c * (size_t)gDIM]);
      if (d2 < best) { best = d2; bestc = c; }
    }

    label[i] = bestc;
    cnt[bestc]++;

    double* sc = &sum[bestc * (size_t)gDIM];
    for (int d = 0; d < gDIM; ++d) sc[d] += pi[d];
  }
}

//hamerly
static void assign_hamerly(const vector<double>& P,
                           const vector<double>& C,
                           const vector<double>& /*prevC*/,
                           const vector<double>& s,
                           const vector<double>& cMove,
                           int K,
                           vector<int>& label,
                           vector<double>& upper,
                           vector<double>& lower,
                           vector<double>& sum,
                           vector<int>& cnt) {
  const int n = (int)label.size();
  std::fill(sum.begin(), sum.end(), 0.0);
  std::fill(cnt.begin(), cnt.end(), 0);

  double maxMove = 0.0;
  for (int c = 0; c < K; ++c) maxMove = std::max(maxMove, cMove[c]);

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    if (a >= 0) upper[i] += cMove[a];
    lower[i] = std::max(0.0, lower[i] - maxMove);
  }

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    bool need = (a < 0);
    if (!need) need = !(upper[i] <= s[a] && upper[i] <= lower[i]);

    double u = upper[i];
    int bestc = a;
    double best = u;
    double second = std::numeric_limits<double>::infinity();
    const double* pi = &P[i * (size_t)gDIM];

    if (need) {
      if (a >= 0) best = std::sqrt(sqdist2(pi, &C[a * (size_t)gDIM]));
      else best = std::numeric_limits<double>::infinity();

      for (int c = 0; c < K; ++c) {
        if (c == a) continue;
        if (a >= 0) {
          double ccsep = l2(&C[a * (size_t)gDIM], &C[c * (size_t)gDIM]);
          if (best <= 0.5 * ccsep && best <= lower[i]) continue;
        }
        double d = std::sqrt(sqdist2(pi, &C[c * (size_t)gDIM]));
        if (d < best) { second = best; best = d; bestc = c; }
        else if (d < second) { second = d; }
      }

      u = best;
      label[i] = bestc;
      upper[i] = u;
      lower[i] = second;
    }

    double* sc = &sum[bestc * (size_t)gDIM];
    for (int d = 0; d < gDIM; ++d) sc[d] += pi[d];
    cnt[bestc]++;
  }
}

//elkan
static void assign_elkan(const vector<double>& P,
                         const vector<double>& C,
                         const vector<double>& /*prevC*/,
                         const vector<double>& cMove,
                         const vector<double>& Dcc,
                         int K,
                         vector<int>& label,
                         vector<double>& upper,
                         vector<double>& lowerMat,
                         vector<double>& sum,
                         vector<int>& cnt) {
  const int n = (int)label.size();
  std::fill(sum.begin(), sum.end(), 0.0);
  std::fill(cnt.begin(), cnt.end(), 0);

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    if (a >= 0) upper[i] += cMove[a];
    double* Li = &lowerMat[i * (size_t)K];
    for (int c = 0; c < K; ++c) Li[c] = std::max(0.0, Li[c] - cMove[c]);
  }

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    double* Li = &lowerMat[i * (size_t)K];
    const double* pi = &P[i * (size_t)gDIM];

    if (a < 0) {
      double best = std::numeric_limits<double>::infinity();
      int bestc = 0;
      double second = std::numeric_limits<double>::infinity();

      for (int c = 0; c < K; ++c) {
        double d = std::sqrt(sqdist2(pi, &C[c * (size_t)gDIM]));
        Li[c] = d;
        if (d < best) { second = best; best = d; bestc = c; }
        else if (d < second) { second = d; }
      }

      label[i] = a = bestc;
      upper[i] = best;

      double* sc = &sum[a * (size_t)gDIM];
      for (int d = 0; d < gDIM; ++d) sc[d] += pi[d];
      cnt[a]++;
      continue;
    }

    const double u0 = upper[i];
    double bound = std::numeric_limits<double>::infinity();
    for (int c = 0; c < K; ++c) {
      if (c == a) continue;
      double gate = std::max(Li[c], 0.5 * Dcc[a * (size_t)K + c]);
      if (gate < bound) bound = gate;
    }

    bool reassess = !(u0 <= bound);
    int bestc = a;
    double best = u0;

    if (reassess) {
      best = std::sqrt(sqdist2(pi, &C[a * (size_t)gDIM]));
      for (int c = 0; c < K; ++c) {
        if (c == a) continue;
        double gate = std::max(Li[c], 0.5 * Dcc[a * (size_t)K + c]);
        if (best <= gate) continue;

        double d = std::sqrt(sqdist2(pi, &C[c * (size_t)gDIM]));
        Li[c] = d;
        if (d < best) { Li[a] = best; best = d; bestc = c; a = c; }
      }
      upper[i] = best;
      label[i] = bestc;
      Li[bestc] = best;
    }

    double* sc = &sum[label[i] * (size_t)gDIM];
    for (int d = 0; d < gDIM; ++d) sc[d] += pi[d];
    cnt[label[i]]++;
  }
}

//yinyang
static void kmeans_group_centroids(const vector<double>& C, int K, int G, vector<int>& gid) {
  G = std::max(1, std::min(G, K));
  gid.assign(K, 0);
  if (G == 1) { std::fill(gid.begin(), gid.end(), 0); return; }

  vector<double> GC((size_t)G * gDIM, 0.0);

  for (int g = 0; g < G; ++g) {
    int idx = (int)std::floor((double)g * K / G);
    for (int d = 0; d < gDIM; ++d) GC[g * (size_t)gDIM + d] = C[idx * (size_t)gDIM + d];
  }

  const int iters = 5;
  vector<int> gcnt(G, 0);

  for (int it = 0; it < iters; ++it) {
    std::fill(gid.begin(), gid.end(), 0);

    for (int c = 0; c < K; ++c) {
      double best = std::numeric_limits<double>::infinity();
      int bestg = 0;
      for (int g = 0; g < G; ++g) {
        double d2 = sqdist2(&C[c * (size_t)gDIM], &GC[g * (size_t)gDIM]);
        if (d2 < best) { best = d2; bestg = g; }
      }
      gid[c] = bestg;
    }

    std::fill(GC.begin(), GC.end(), 0.0);
    std::fill(gcnt.begin(), gcnt.end(), 0);
    for (int c = 0; c < K; ++c) {
      int g = gid[c];
      for (int d = 0; d < gDIM; ++d) GC[g * (size_t)gDIM + d] += C[c * (size_t)gDIM + d];
      gcnt[g]++;
    }
    for (int g = 0; g < G; ++g) if (gcnt[g] > 0)
      for (int d = 0; d < gDIM; ++d) GC[g * (size_t)gDIM + d] /= gcnt[g];
  }
}

static void assign_yinyang(const vector<double>& P,
                           const vector<double>& C,
                           const vector<double>& /*prevC*/,
                           const vector<int>& gid,
                           int G,
                           const vector<double>& cMove,
                           const vector<double>& /*Dcc*/,
                           int K,
                           vector<int>& label,
                           vector<double>& upper,
                           vector<double>& lowerG,
                           vector<double>& sum,
                           vector<int>& cnt) {
  const int n = (int)label.size();
  std::fill(sum.begin(), sum.end(), 0.0);
  std::fill(cnt.begin(), cnt.end(), 0);

  vector<double> gMove(G, 0.0);
  for (int c = 0; c < K; ++c) gMove[gid[c]] = std::max(gMove[gid[c]], cMove[c]);

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    if (a >= 0) upper[i] += cMove[a];
    double* LG = &lowerG[i * (size_t)G];
    for (int g = 0; g < G; ++g) LG[g] = std::max(0.0, LG[g] - gMove[g]);
  }

  for (int i = 0; i < n; ++i) {
    int a = label[i];
    const double* pi = &P[i * (size_t)gDIM];
    double u = upper[i];
    double* LG = &lowerG[i * (size_t)G];

    double minLG = std::numeric_limits<double>::infinity();
    for (int g = 0; g < G; ++g) minLG = std::min(minLG, LG[g]);

    bool need = (a < 0) || !(u <= minLG);
    double best = u;
    int bestc = a;

    if (need) {
      if (a >= 0) best = std::sqrt(sqdist2(pi, &C[a * (size_t)gDIM]));
      else best = std::numeric_limits<double>::infinity();

      for (int g = 0; g < G; ++g) {
        if (LG[g] >= best) continue;
        for (int c = 0; c < K; ++c) {
          if (gid[c] != g || c == a) continue;
          double d = std::sqrt(sqdist2(pi, &C[c * (size_t)gDIM]));
          if (d < best) { best = d; bestc = c; }
        }
      }

      for (int g = 0; g < G; ++g) {
        double gbest = std::numeric_limits<double>::infinity();
        for (int c = 0; c < K; ++c) {
          if (gid[c] != g) continue;
          double d = std::sqrt(sqdist2(pi, &C[c * (size_t)gDIM]));
          if (d < gbest) gbest = d;
        }
        LG[g] = gbest;
      }

      a = bestc;
      u = best;
      label[i] = a;
      upper[i] = u;
    }

    double* sc = &sum[a * (size_t)gDIM];
    for (int d = 0; d < gDIM; ++d) sc[d] += pi[d];
    cnt[a]++;
  }
}

// parse CLI args (--k, --limit, --cols)
struct Parsed {
  int         K            = 5;
  long long   limit        = -1;
  vector<int> cols;
  bool        colsSpecified = false;
};

static Parsed parse_args(int argc, char** argv) {
  std::unordered_map<string, int> cmap{
      {"GAP", 2}, {"GRP", 3}, {"V", 4}, {"GI", 5}, {"SM1", 6}, {"SM2", 7}, {"SM3", 8},
  };

  Parsed a;

  for (int i = 1; i < argc; ++i) {
    string s(argv[i]);

    auto take_val = [&](const string&) -> string {
      size_t p = s.find('=');
      if (p == string::npos) return "";
      return s.substr(p + 1);
    };

    if (s.rfind("--k=", 0) == 0) {
      a.K = std::stoi(take_val("--k"));
      if (a.K <= 0) throw std::runtime_error("--k must be > 0");
    } else if (s.rfind("--limit=", 0) == 0) {
      a.limit = std::stoll(take_val("--limit"));
    } else if (s.rfind("--cols=", 0) == 0) {
      string v = take_val("--cols");
      a.cols.clear();
      size_t start = 0;
      while (true) {
        size_t pos = v.find(',', start);
        string name = (pos == string::npos) ? v.substr(start) : v.substr(start, pos - start);
        if (!name.empty()) {
          auto it = cmap.find(name);
          if (it == cmap.end()) throw std::runtime_error("Unknown column: " + name);
          a.cols.push_back(it->second);
        }
        if (pos == string::npos) break;
        start = pos + 1;
      }
      a.colsSpecified = true;
    } else {
      std::cerr << "Warning: unknown arg ignored: " << s << "\n";
    }
  }

  if (!a.colsSpecified) {
    a.cols = {2, 3, 4, 5, 6, 7, 8};
    a.colsSpecified = true;
  }

  return a;
}


//main
int main(int argc, char** argv) {
  Parsed args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Arg error: " << e.what() << "\n";
    return 1;
  }

  int K = args.K;
  long long limit = args.limit;
  int DIM = (int)args.cols.size();
  gDIM = DIM;

  if (DIM <= 0) {
    std::cerr << "No features selected (DIM=0). Use --cols from {GAP,GRP,V,GI,SM1,SM2,SM3}.\n";
    return 1;
  }

  vector<int> cols = args.cols;

  std::cerr << "K=" << K << " | DIM=" << DIM << " | cols:";
  for (int c : cols) std::cerr << ' ' << c;
  std::cerr << " | limit=" << limit << "\n";

  int choice = 1;
  {
    cout << "Choose k-means variant:\n"
            " [1] Lloyd\n"
            " [2] Elkan\n"
            " [3] Hamerly\n"
            " [4] Yinyang\n"
            "Enter choice: ";
    cin >> choice;
    if (choice < 1 || choice > 4) choice = 1;
  }

  vector<double> P;
  long long N = 0;

  double t_io  = 0.0;
  auto   t_io0 = clk::now();

  try {
    fs::path fpath = fs::path(__FILE__).parent_path().parent_path() / "household_power_consumption.txt";
    std::ifstream ifs(fpath);
    if (!ifs) throw std::runtime_error("Cannot open file: " + fpath.string());

    string line;
    if (!std::getline(ifs, line)) throw std::runtime_error("Empty file");

    auto oknum = [](std::string s) -> bool {
      trim_inplace(s);
      if (s.empty() || s == "?") return false;
      char* end = nullptr;
      std::strtod(s.c_str(), &end);
      return end && *end == '\0';
    };

    P.reserve(1 << 20);

    while (std::getline(ifs, line)) {
      if (limit > 0 && N >= limit) break;

      vector<string> tok;
      tok.reserve(10);
      size_t start = 0;
      while (true) {
        size_t pos = line.find(';', start);
        if (pos == string::npos) { tok.emplace_back(line.substr(start)); break; }
        tok.emplace_back(line.substr(start, pos - start));
        start = pos + 1;
      }

      if ((int)tok.size() < 9) continue;

      bool good = true;
      for (int id : cols) {
        if (id >= (int)tok.size()) { good = false; break; }
        trim_inplace(tok[id]);
        if (!oknum(tok[id])) { good = false; break; }
      }
      if (!good) continue;

      for (int id : cols) P.push_back(std::stod(tok[id]));
      ++N;
    }

    ifs.close();

    if (N < K) {
      std::cerr << "Not enough rows after cleaning: N=" << N << " < K=" << K << "\n";
      return 1;
    }

    auto t_io1 = clk::now();
    t_io = secd(t_io1 - t_io0).count();

  } catch (const std::exception& e) {
    std::cerr << "Read error: " << e.what() << "\n";
    return 1;
  }

  vector<double> C((size_t)K * DIM), prevC((size_t)K * DIM);
  {
    std::mt19937 gen(2025);
    vector<long long> idx(N);
    std::iota(idx.begin(), idx.end(), 0LL);
    std::shuffle(idx.begin(), idx.end(), gen);
    for (int i = 0; i < K; ++i) {
      long long id = idx[i];
      for (int d = 0; d < DIM; ++d) C[i * (size_t)DIM + d] = P[id * (size_t)DIM + d];
    }
    prevC = C;
  }

  vector<int>    label((size_t)N, -1);
  vector<double> sum((size_t)K * DIM, 0.0);
  vector<int>    cnt(K, 0);

  vector<double> upper(N, std::numeric_limits<double>::infinity());
  vector<double> lower(N, 0.0);
  vector<double> cMove(K, 0.0), sK(K, 0.0);
  vector<double> Dcc;
  vector<double> lowerMat;
  vector<double> lowerG;
  int            G = std::max(1, std::min(K, 10));
  vector<int>    gid(K, 0);

  if (choice == 2) lowerMat.assign((size_t)N * K, 0.0);
  if (choice == 4) lowerG.assign((size_t)N * G, 0.0);

  const int    max_iter  = 100;
  const double tol       = 1e-4;
  int          it        = 0;
  double       max_shift = 0.0;

  auto t_comp0 = clk::now();

  for (it = 0; it < max_iter; ++it) {
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(cnt.begin(), cnt.end(), 0);

    for (int c = 0; c < K; ++c) {
      double sh = 0.0;
      for (int d = 0; d < DIM; ++d) {
        double dx = C[c * (size_t)DIM + d] - prevC[c * (size_t)DIM + d];
        sh += dx * dx;
      }
      cMove[c] = std::sqrt(sh);
    }
    if (choice == 3) compute_center_separation(C, K, sK);
    if (choice == 2 || choice == 4) compute_center_dists(C, K, Dcc);
    if (choice == 4) kmeans_group_centroids(C, K, G, gid);

    // assignment
    switch (choice) {
      case 1: assign_lloyd(P, C, K, label, sum, cnt); break;
      case 2: assign_elkan(P, C, prevC, cMove, Dcc, K, label, upper, lowerMat, sum, cnt); break;
      case 3: assign_hamerly(P, C, prevC, sK, cMove, K, label, upper, lower, sum, cnt); break;
      case 4: assign_yinyang(P, C, prevC, gid, G, cMove, Dcc, K, label, upper, lowerG, sum, cnt); break;
    }

    // update
    prevC = C;
    max_shift = 0.0;

    for (int c = 0; c < K; ++c) {
      double sh = 0.0;
      if (cnt[c] > 0) {
        for (int d = 0; d < DIM; ++d) {
          double oldv = C[c * (size_t)DIM + d];
          C[c * (size_t)DIM + d] = sum[c * (size_t)DIM + d] / cnt[c];
          double dx = C[c * (size_t)DIM + d] - oldv;
          sh += dx * dx;
        }
      } else {
        for (int d = 0; d < DIM; ++d) { double dx = 0.0; sh += dx * dx; }
      }
      double mv = std::sqrt(sh);
      if (mv > max_shift) max_shift = mv;
    }

    if (max_shift < tol) break;
  }

  auto   t_comp1   = clk::now();
  double t_compute = secd(t_comp1 - t_comp0).count();

  //report
  double t_total = t_io + t_compute;
  double p       = 1.0 - (t_io / t_total);

  cout << "--------------------------------------------------------------------------\n";
  cout << "Converged after " << (it + 1) << " iteration(s)\n";
  cout << "Variant: " << (choice == 1 ? "Lloyd" : choice == 2 ? "Elkan" : choice == 3 ? "Hamerly" : "Yinyang") << "\n";
  cout << "I/O time (s): " << t_io << "\n";
  cout << "Compute time (s): " << t_compute << "\n";
  cout << "Total time (s): " << t_total << "\n";
  cout << "Parallelizable fraction p ≈ " << p << "\n";
  cout << "Rows used (N): " << N << " | Features (DIM): " << DIM << " | K: " << K << "\n";

  //output
  {
    fs::path out = fs::current_path() / "clustering_result.txt";
    std::ofstream ofs(out);
    ofs << std::fixed << std::setprecision(3);

    for (int c = 0; c < K; ++c) {
      ofs << "Cluster " << (c + 1) << " : (";
      for (int d = 0; d < DIM; ++d) { if (d) ofs << ", "; ofs << C[c * (size_t)DIM + d]; }
      ofs << ") --> ";

      bool first = true;
      for (long long i = 0; i < N; ++i) {
        if (label[(size_t)i] != c) continue;
        if (!first) ofs << ", ";
        first = false;
        ofs << "(";
        for (int d = 0; d < DIM; ++d) { if (d) ofs << ", "; ofs << P[(size_t)i * (size_t)DIM + d]; }
        ofs << ")";
      }
      ofs << "\n";
    }

    ofs.close();
    cout << "Wrote " << out << "\n";
  }

  return 0;
}
