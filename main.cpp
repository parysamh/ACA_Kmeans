#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>

using namespace std;

const int DIM = 2;

float distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0;
    for (int i = 0; i < DIM; ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

void assign_labels(int N, int K, const vector<vector<float>>& points,
                   const vector<vector<float>>& centroids, vector<int>& labels) {
    for (int i = 0; i < N; ++i) {
        float min_dist = distance(points[i], centroids[0]);
        int min_index = 0;
        for (int j = 1; j < K; ++j) {
            float d = distance(points[i], centroids[j]);
            if (d < min_dist) {
                min_dist = d;
                min_index = j;
            }
        }
        labels[i] = min_index;
    }
}

void update_centroids(int N, int K, const vector<vector<float>>& points,
                      vector<vector<float>>& centroids, const vector<int>& labels) {
    vector<vector<float>> sum(K, vector<float>(DIM, 0.0));
    vector<int> count(K, 0);

    for (int i = 0; i < N; ++i) {
        int label = labels[i];
        for (int j = 0; j < DIM; ++j)
            sum[label][j] += points[i][j];
        count[label]++;
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < DIM; ++j) {
            if (count[i] > 0)
                centroids[i][j] = sum[i][j] / count[i];
        }
    }
}

int main() {
    int N, K;
    cout << "Enter number of points: ";
    cin >> N;
    cout << "Enter number of clusters (K): ";
    cin >> K;

    vector<vector<float>> points(N, vector<float>(DIM));
    vector<vector<float>> centroids(K, vector<float>(DIM));
    vector<int> labels(N);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0);
    uniform_int_distribution<> idx_dis(0, N - 1);

    // Generate random points
    for (int i = 0; i < N; ++i) {
        points[i][0] = dis(gen);
        points[i][1] = dis(gen);
    }

    // Choose random initial centroids
    for (int i = 0; i < K; ++i) {
        int idx = idx_dis(gen);
        centroids[i] = points[idx];
    }

    // Run K-Means
    for (int iter = 0; iter < 10; ++iter) {
        assign_labels(N, K, points, centroids, labels);
        update_centroids(N, K, points, centroids, labels);
    }

    // Write result to file
    ofstream outfile("../clustering_result.txt");
    if (!outfile.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return 1;
    }

    outfile << fixed << setprecision(1);
    for (int c = 0; c < K; ++c) {
        outfile << "Cluster " << c << " : ("
                << centroids[c][0] << ", " << centroids[c][1] << ") ---> ";

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
    cout << "\nClustering result written to 'clustering_result.txt'\n";

    return 0;
}
