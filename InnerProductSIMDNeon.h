// 对内积计算进行SIMD优化

#pragma once
#include <queue>


#include <arm_neon.h>

// SIMD 内积计算
float InnerProductSIMDNeon(float* base, float* query, int vecdim) {
    float32x4_t sum = vmovq_n_f32(0.0f);  // 初始化为零，存储 4 个浮点数
    for (int i = 0; i < vecdim; i += 4) {  // 每次处理 4 个浮点数
        float32x4_t va = vld1q_f32(base + i);  // 从 base 加载 4 个浮点数
        float32x4_t vb = vld1q_f32(query + i);  // 从 query 加载 4 个浮点数
        sum = vmlaq_f32(sum, va, vb);  // 对应元素相乘并相加
    }

    // 将 NEON 向量寄存器中的结果存储到数组
    float tmp[4];
    vst1q_f32(tmp, sum);

    // 将 4 个浮点数加起来并返回 1 - 内积
    return 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
}


std::priority_queue<std::pair<float, uint32_t>> flat_search_InnerProductSIMD(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < base_number; ++i) {
        // 使用 SIMD 内积计算
        float dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);

        // 计算 1 - 内积距离
        // dis = 1 - dis;

        // 更新优先队列
        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }

    return q;
}
