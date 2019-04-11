#pragma once

#include <memory>

#include <torch/torch.h>

#include "cpprl/distributions/distribution.h"

using namespace torch;

namespace cpprl
{
class OutputLayer : public nn::Module
{
  public:
    virtual ~OutputLayer() = 0;

    virtual std::unique_ptr<Distribution> forward(torch::Tensor x) = 0;
};

inline OutputLayer::~OutputLayer() {}

class CategoricalOutput : public OutputLayer
{
  private:
    nn::Linear linear;

  public:
    CategoricalOutput(unsigned int num_inputs, unsigned int num_outputs);

    std::unique_ptr<Distribution> forward(torch::Tensor x);
};
}