#pragma once

#include <vector>

#include <torch/torch.h>

#include "cpprl/model/nn_base.h"

using namespace torch;

namespace cpprl
{
class CnnBase : public NNBase
{
  private:
    nn::Sequential main;
    nn::Sequential critic_linear;

  public:
    CnnBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 512);

    std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                       torch::Tensor hxs,
                                       torch::Tensor masks);
};
}