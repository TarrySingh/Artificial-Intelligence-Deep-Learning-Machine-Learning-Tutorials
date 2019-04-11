#pragma once

#include <vector>

#include <torch/torch.h>

#include "cpprl/model/nn_base.h"

using namespace torch;

namespace cpprl
{
class MlpBase : public NNBase
{
  private:
    nn::Sequential actor;
    nn::Sequential critic;
    nn::Linear critic_linear;

  public:
    MlpBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 64);

    std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                       torch::Tensor hxs,
                                       torch::Tensor masks);
};
}