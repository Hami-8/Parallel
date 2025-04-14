#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <cassert>
#include <arm_neon.h>
#include "pq_index.h"



// ----------------------
// (1) 利用 NEON 加速构建单个子空间的 LUT
// 对于查询子向量 query_sub（长度 sub_dim），计算其与该子空间所有 256 个聚类中心的内积距离，并存入 LUT 数组。
// 距离公式：LUT[c] = 1 - dot(query_sub, center_c)
void build_LUT_for_subspace(const float* query_sub, const Codebook &codebook, float* LUT_array) {
    int sub_dim = codebook.dim;        // 如24
    int num_clusters = codebook.clusters; // 如256
    // 遍历每个聚类中心
    for (int c = 0; c < num_clusters; c++) {
        const float* center = codebook.centers[c].data();
        float32x4_t sum_vec = vmovq_n_f32(0.0f);
        int j = 0;
        // 以 4 个 float 为一组处理，假设 sub_dim 可整除4（例如24）
        for (; j <= sub_dim - 4; j += 4) {
            float32x4_t vq = vld1q_f32(query_sub + j);
            float32x4_t vc = vld1q_f32(center + j);
            sum_vec = vmlaq_f32(sum_vec, vq, vc);
        }
        // 累加 sum_vec 中 4 个元素
        float sum_arr[4];
        vst1q_f32(sum_arr, sum_vec);
        float dot = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        LUT_array[c] = 1.0f - dot;  // 计算内积距离
    }
}

// ----------------------
// (2) 在线查询函数：利用 PQ 索引快速查表计算距离（结合 NEON 构建 LUT 部分）
//
// 参数说明：
//   query: 查询向量，长度为 vecdim（例如96）
//   pq_codes: 每个 base 向量的 PQ 编码，每个编码为一个 vector<uint8_t>，长度等于子空间数（例如4）
//   codebooks: 每个子空间的 Codebook，大小等于子空间数
//   base_number: 数据库中向量的数量
//   vecdim: 总维度（例如96）
//   k: 需要返回的最近邻数
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    const float* query,
    const std::vector<std::vector<uint8_t>> &pq_codes,
    const std::vector<Codebook> &codebooks,
    size_t base_number, size_t vecdim, size_t k)
{
    // 子空间数目由 codebooks.size() 得出
    int subspace_num = codebooks.size();
    assert(vecdim % subspace_num == 0);
    int sub_dim = vecdim / subspace_num;

    // 为每个子空间构建 LUT：LUT[s] 大小为聚类数（如256）
    std::vector<std::vector<float>> LUT(subspace_num, std::vector<float>(codebooks[0].clusters, 0.0f));
    for (int s = 0; s < subspace_num; s++) {
        const float* query_sub = query + s * sub_dim;
        // 调用 NEON 加速函数构建 LUT for 子空间 s
        build_LUT_for_subspace(query_sub, codebooks[s], LUT[s].data());
    }

    // 使用 PQ 查表计算每个 base 向量与查询向量的距离
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < base_number; i++) {
        float dist = 0.0f;
        // 每个 base 向量的 PQ 编码存储在 pq_codes[i]（长度为 subspace_num）
        // 累加每个子空间对应的查表距离
        for (int s = 0; s < subspace_num; s++) {
            uint8_t code = pq_codes[i][s];  // 取出第 s 个子空间的聚类中心编号
            dist += LUT[s][code];
        }
        // 更新优先队列，保持 top-k 最小距离的向量
        if (q.size() < k) {
            q.push({dist, i});
        } else {
            if (dist < q.top().first) {
                q.push({dist, i});
                q.pop();
            }
        }
    }
    return q;
}
