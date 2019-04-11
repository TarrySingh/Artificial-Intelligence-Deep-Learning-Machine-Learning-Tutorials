#pragma once

#include <torch/torch.h>

#include "cpprl/generators/generator.h"

namespace cpprl
{
class FeedForwardGenerator : public Generator
{
  private:
    torch::Tensor observations, hidden_states, actions, value_predictions,
        returns, masks, action_log_probs, advantages, indices;
    int index;

  public:
    FeedForwardGenerator(int mini_batch_size,
                         torch::Tensor observations,
                         torch::Tensor hidden_states,
                         torch::Tensor actions,
                         torch::Tensor value_predictions,
                         torch::Tensor returns,
                         torch::Tensor masks,
                         torch::Tensor action_log_probs,
                         torch::Tensor advantages);

    virtual bool done() const;
    virtual MiniBatch next();
};
}