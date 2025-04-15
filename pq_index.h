#pragma once
#include <cstdint>
#include <iostream>
#include "kmeans.h"
#include "faiss/impl/ProductQuantizer.h"
#include <faiss/IndexPQ.h>

#include <cmath>

// 对一个向量进行归一化，vec 为指向向量的指针，dim 为维度
void normalize_vector(float* vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    // 防止除零
    if (norm > 1e-6) {
        for (int i = 0; i < dim; i++) {
            vec[i] /= norm;
        }
    }
}

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


// 离线 PQ 索引构建函数
// 参数说明：
//   base: 数据库向量数组，大小为 base_number * dim
//   base_number: 数据库向量个数
//   dim: 向量维度（本实验中为 96）
//   save_path: 保存索引（包含 PQ 模型和编码数据）的文件路径
void build_pq_index(float* base, size_t base_number, size_t dim, const std::string &save_path) {
    // 设置 PQ 参数：将向量划分为 4 个子空间，每个子空间用 8 位编码
    int m = 4;         // 子量化器个数
    int nbits = 8;     // 每个子空间每个编码的位数
    std::cout << "Initializing ProductQuantizer: dim=" << dim << ", m=" << m << ", nbits=" << nbits << std::endl;

    // 构造 ProductQuantizer 对象（FAISS 内部会自动划分子空间）
    faiss::ProductQuantizer pq(dim, m, nbits);

    // 训练 PQ 模型，训练数据为数据库向量
    std::cout << "Training PQ on " << base_number << " vectors ..." << std::endl;
    pq.train(base_number, base);
    std::cout << "PQ training complete." << std::endl;

    // 分配存储 PQ 编码的数组，每个向量产生 m 个 8 位编码，共 base_number * m 字节
    std::vector<uint8_t> codes(base_number * m);
    
    // 对数据库向量进行编码，生成 PQ 编码
    pq.compute_codes(base, base_number, codes.data());
    std::cout << "PQ codes computed for all vectors." << std::endl;

    // 将 PQ 模型和编码数据保存到磁盘
    std::ofstream ofs(save_path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open file " << save_path << " for writing." << std::endl;
        return;
    }
    // 保存 PQ 模型（包括各子空间的聚类中心）
    pq.write(ofs);
    // 保存 PQ 编码数据：连续 base_number * m 字节
    ofs.write(reinterpret_cast<const char*>(codes.data()), codes.size() * sizeof(uint8_t));
    ofs.close();
    std::cout << "PQ index saved to " << save_path << std::endl;
}