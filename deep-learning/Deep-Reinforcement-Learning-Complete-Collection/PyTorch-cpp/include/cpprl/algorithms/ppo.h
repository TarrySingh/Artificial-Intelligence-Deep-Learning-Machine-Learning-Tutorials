#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "cpprl/algorithms/algorithm.h"

namespace cpprl
{
class Policy;
class ROlloutStorage;

class PPO : public Algorithm
{
  private:
    Policy &policy;
    float clip_param, value_loss_coef, entropy_coef, max_grad_norm;
    int num_epoch, num_mini_batch;
    std::unique_ptr<torch::optim::Optimizer> optimizer;

  public:
    PPO(Policy &policy,
        float clip_param,
        int num_epoch,
        int num_mini_batch,
        float value_loss_coef,
        float entropy_coef,
        float learning_rate,
        float epsilon = 1e-8,
        float max_grad_norm = 0.5);

    std::vector<UpdateDatum> update(RolloutStorage &rollouts);
};
}