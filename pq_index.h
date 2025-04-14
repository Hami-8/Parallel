#pragma once
#include <cstdint>
#include <iostream>
#include "kmeans.h"

// 构建 PQ 索引
// 参数说明：
//   base: 原始数据库向量数组，大小为 base_number * vecdim
//   base_number: 数据库中向量数量
//   vecdim: 向量维度（例如96）
//   subspace_num: 子空间数目（例如4）
//   clusters_per_subspace: 每个子空间的聚类数（例如256）
// 输出：
//   codes: 每个向量的 PQ 编码，每个编码由 subspace_num 个 uint8_t 组成
//   codebooks: 每个子空间的 codebook，包含聚类中心
void build_PQ_index(float* base, size_t base_number, size_t vecdim,
                    int subspace_num, int clusters_per_subspace,
                    std::vector<std::vector<uint8_t>> &codes,
                    std::vector<Codebook> &codebooks) {
    // 检查向量维度能否均分
    assert(vecdim % subspace_num == 0);
    int sub_dim = vecdim / subspace_num;

    // 初始化 codebooks 的大小（每个子空间一个）
    codebooks.resize(subspace_num);
    // 为每个子空间构造一个二维数据数组：每个 base 向量在该子空间的子向量
    std::vector<std::vector<std::vector<float>>> subspace_data(
        subspace_num, std::vector<std::vector<float>>(base_number, std::vector<float>(sub_dim)));

    // 按子空间划分 base 中的每个向量
    for (size_t i = 0; i < base_number; i++) {
        for (int s = 0; s < subspace_num; s++) {
            for (int j = 0; j < sub_dim; j++) {
                subspace_data[s][i][j] = base[i * vecdim + s * sub_dim + j];
            }
        }
    }

    // 对每个子空间分别运行 k-means 得到 codebook
    int max_iter = 20;  // 最大迭代次数，可根据需求调整
    // assignments[s] 保存第 s 个子空间每个向量的聚类编号
    std::vector<std::vector<int>> assignments(subspace_num);
    for (int s = 0; s < subspace_num; s++) {
        run_kmeans_for_subspace(subspace_data[s], clusters_per_subspace, max_iter, codebooks[s], assignments[s]);
        std::cout << "Subspace " << s << " k-means finished." << std::endl;
    }

    // 编码：为每个 base 向量生成 PQ 码（每个码由 subspace_num 个 uint8_t 组成）
    codes.resize(base_number, std::vector<uint8_t>(subspace_num));
    for (size_t i = 0; i < base_number; i++) {
        for (int s = 0; s < subspace_num; s++) {
            codes[i][s] = static_cast<uint8_t>(assignments[s][i]);
        }
    }
}
