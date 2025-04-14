#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cassert>

// 用于保存单个子空间的 codebook
struct Codebook {
    int clusters;                     // 聚类个数（通常为256）
    int dim;                          // 子空间维度（例如 24）
    std::vector<std::vector<float>> centers; // 每个聚类中心：[cluster_idx][维度]
};

// 使用简单 k-means 算法处理一个子空间数据
// sub_data：二维数组，每个元素为一个子向量，维度为 sub_dim
// clusters：聚类数目（例如256）
// max_iter：最大迭代次数
// codebook：输出的聚类中心（即 codebook）
// assignments：输出的每个样本所属的聚类编号
void run_kmeans_for_subspace(const std::vector<std::vector<float>> &sub_data,
                             int clusters, int max_iter,
                             Codebook &codebook, std::vector<int> &assignments) {
    int N = sub_data.size();
    assert(N > 0);
    int d = sub_data[0].size(); // 子空间维度
    codebook.dim = d;
    codebook.clusters = clusters;
    codebook.centers.resize(clusters, std::vector<float>(d, 0.0f));

    // 随机初始化 centers：从数据集中随机选取 clusters 个样本
    for (int c = 0; c < clusters; c++) {
        int idx = rand() % N;
        codebook.centers[c] = sub_data[idx];
    }

    assignments.assign(N, 0);

    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        // 分配每个样本到最近的聚类中心
        for (int i = 0; i < N; i++) {
            float best_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < clusters; c++) {
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float diff = sub_data[i][j] - codebook.centers[c][j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if (best_c != assignments[i]) {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if (!changed) break;  // 如果所有样本的分类均未改变，则退出迭代

        // 更新聚类中心
        std::vector<std::vector<float>> sum(clusters, std::vector<float>(d, 0.0f));
        std::vector<int> count(clusters, 0);
        for (int i = 0; i < N; i++) {
            int c = assignments[i];
            for (int j = 0; j < d; j++) {
                sum[c][j] += sub_data[i][j];
            }
            count[c]++;
        }
        for (int c = 0; c < clusters; c++) {
            if (count[c] > 0) {
                for (int j = 0; j < d; j++) {
                    codebook.centers[c][j] = sum[c][j] / count[c];
                }
            }
        }
    }
}
