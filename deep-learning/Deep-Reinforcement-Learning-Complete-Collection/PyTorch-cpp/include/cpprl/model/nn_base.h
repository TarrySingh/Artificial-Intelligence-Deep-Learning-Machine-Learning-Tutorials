#pragma once

#include <vector>

#include <torch/torch.h>

using namespace torch;

namespace cpprl
{
class NNBase : public nn::Module
{
  private:
    bool recurrent;
    unsigned int hidden_size;
    nn::GRU gru;

  public:
    NNBase(bool recurrent,
           unsigned int recurrent_input_size,
           unsigned int hidden_size);

    virtual std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                               torch::Tensor hxs,
                                               torch::Tensor masks);
    std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                           torch::Tensor hxs,
                                           torch::Tensor masks);
    unsigned int get_hidden_size() const;

    inline int get_output_size() const { return hidden_size; }
    inline bool is_recurrent() const { return recurrent; }
};
}