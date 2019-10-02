#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "cpprl/algorithms/algorithm.h"

namespace cpprl
{
class Policy;
class ROlloutStorage;

class A2C : public Algorithm
{
  private:
    Policy &policy;
    float value_loss_coef, entropy_coef, max_grad_norm;
    std::unique_ptr<torch::optim::Optimizer> optimizer;

  public:
    A2C(Policy &policy,
        float value_loss_coef,
        float entropy_coef,
        float learning_rate,
        float epsilon = 1e-8,
        float alpha = 0.99,
        float max_grad_norm = 0.5);

    std::vector<UpdateDatum> update(RolloutStorage &rollouts);
};
}