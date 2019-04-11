#pragma once

#include <memory>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include "cpprl/generators/generator.h"
#include "cpprl/spaces.h"

namespace cpprl
{
class RolloutStorage
{
  private:
    torch::Tensor observations, hidden_states, rewards, value_predictions,
        returns, action_log_probs, actions, masks;
    torch::Device device;
    unsigned int num_steps;
    unsigned int step;

  public:
    RolloutStorage(unsigned int num_steps,
                   unsigned int num_processes,
                   c10::ArrayRef<int64_t> obs_shape,
                   ActionSpace action_space,
                   unsigned int hidden_state_size,
                   torch::Device device);

    void after_update();
    void compute_returns(torch::Tensor next_value,
                         bool use_gae,
                         float gamma,
                         float tau);
    std::unique_ptr<Generator> feed_forward_generator(torch::Tensor advantages,
                                                      int num_mini_batch);
    void insert(torch::Tensor observation,
                torch::Tensor hidden_state,
                torch::Tensor action,
                torch::Tensor action_log_prob,
                torch::Tensor value_prediction,
                torch::Tensor reward,
                torch::Tensor mask);
    std::unique_ptr<Generator> recurrent_generator(torch::Tensor advantages,
                                                   int num_mini_batch);
    void set_first_observation(torch::Tensor observation);
    void to(torch::Device device);

    inline const torch::Tensor &get_actions() const { return actions; }
    inline const torch::Tensor &get_action_log_probs() const { return action_log_probs; }
    inline const torch::Tensor &get_hidden_states() const { return hidden_states; }
    inline const torch::Tensor &get_masks() const { return masks; }
    inline const torch::Tensor &get_observations() const { return observations; }
    inline const torch::Tensor &get_returns() const { return returns; }
    inline const torch::Tensor &get_rewards() const { return rewards; }
    inline const torch::Tensor &get_value_predictions() const
    {
        return value_predictions;
    }
};
}