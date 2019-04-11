#pragma once

#include <torch/torch.h>

namespace cpprl
{
class Distribution
{
  public:
    virtual ~Distribution() = 0;

    virtual torch::Tensor entropy() = 0;
    virtual torch::Tensor get_logits() = 0;
    virtual torch::Tensor get_probs() = 0;
    virtual torch::Tensor log_prob(torch::Tensor value) = 0;
    virtual torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) = 0;
};

inline Distribution::~Distribution() {}
}