#include <algorithm>
#include <vector>

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "cpprl/generators/recurrent_generator.h"
#include "cpprl/generators/generator.h"
#include "third_party/doctest.h"

namespace cpprl
{
torch::Tensor flatten_helper(int timesteps, int processes, torch::Tensor tensor)
{
    auto tensor_shape = tensor.sizes().vec();
    tensor_shape.erase(tensor_shape.begin());
    tensor_shape[0] = timesteps * processes;
    return tensor.view(tensor_shape);
}

RecurrentGenerator::RecurrentGenerator(int num_processes,
                                       int num_mini_batch,
                                       torch::Tensor observations,
                                       torch::Tensor hidden_states,
                                       torch::Tensor actions,
                                       torch::Tensor value_predictions,
                                       torch::Tensor returns,
                                       torch::Tensor masks,
                                       torch::Tensor action_log_probs,
                                       torch::Tensor advantages)
    : observations(observations),
      hidden_states(hidden_states),
      actions(actions),
      value_predictions(value_predictions),
      returns(returns),
      masks(masks),
      action_log_probs(action_log_probs),
      advantages(advantages),
      indices(torch::randperm(num_processes, torch::TensorOptions(torch::kLong))),
      index(0),
      num_envs_per_batch(num_processes / num_mini_batch) {}

bool RecurrentGenerator::done() const
{
    return index >= indices.size(0);
}

MiniBatch RecurrentGenerator::next()
{
    if (index >= indices.size(0))
    {
        spdlog::error("No minibatches left in generator. Index {}, minibatch "
                      "count: {}.",
                      index, indices.size(0));
        throw std::exception();
    }

    MiniBatch mini_batch;

    // Fill minibatch with tensors of shape (timestep, process, *whatever)
    // Except hidden states, that is just (process, *whatever)
    long env_index = indices[index].item().toLong();
    mini_batch.observations = observations
                                  .narrow(0, 0, observations.size(0) - 1)
                                  .narrow(1, env_index, num_envs_per_batch);
    mini_batch.hidden_states = hidden_states[0]
                                   .narrow(0, env_index, num_envs_per_batch)
                                   .view({num_envs_per_batch, -1});
    mini_batch.actions = actions.narrow(1, env_index, num_envs_per_batch);
    mini_batch.value_predictions = value_predictions
                                       .narrow(0, 0, value_predictions.size(0) - 1)
                                       .narrow(1, env_index, num_envs_per_batch);
    mini_batch.returns = returns.narrow(0, 0, returns.size(0) - 1)
                             .narrow(1, env_index, num_envs_per_batch);
    mini_batch.masks = masks.narrow(0, 0, masks.size(0) - 1)
                           .narrow(1, env_index, num_envs_per_batch);
    mini_batch.action_log_probs = action_log_probs.narrow(1, env_index,
                                                          num_envs_per_batch);
    mini_batch.advantages = advantages.narrow(1, env_index, num_envs_per_batch);

    // Flatten tensors to (timestep * process, *whatever)
    int num_timesteps = mini_batch.observations.size(0);
    int num_processes = num_envs_per_batch;
    mini_batch.observations = flatten_helper(num_timesteps, num_processes,
                                             mini_batch.observations);
    mini_batch.actions = flatten_helper(num_timesteps, num_processes,
                                        mini_batch.actions);
    mini_batch.value_predictions = flatten_helper(num_timesteps, num_processes,
                                                  mini_batch.value_predictions);
    mini_batch.returns = flatten_helper(num_timesteps, num_processes,
                                        mini_batch.returns);
    mini_batch.masks = flatten_helper(num_timesteps, num_processes,
                                      mini_batch.masks);
    mini_batch.action_log_probs = flatten_helper(num_timesteps, num_processes,
                                                 mini_batch.action_log_probs);
    mini_batch.advantages = flatten_helper(num_timesteps, num_processes,
                                           mini_batch.advantages);

    index++;

    return mini_batch;
}

TEST_CASE("RecurrentGenerator")
{
    RecurrentGenerator generator(3, 3, torch::rand({6, 3, 4}),
                                 torch::rand({6, 3, 3}), torch::rand({5, 3, 1}), torch::rand({6, 3, 1}), torch::rand({6, 3, 1}), torch::ones({6, 3, 1}), torch::rand({5, 3, 1}), torch::rand({5, 3, 1}));

    SUBCASE("Minibatch tensors are correct sizes")
    {
        auto minibatch = generator.next();

        CHECK(minibatch.observations.sizes().vec() == std::vector<long>{5, 4});
        CHECK(minibatch.hidden_states.sizes().vec() == std::vector<long>{1, 3});
        CHECK(minibatch.actions.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.value_predictions.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.returns.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.masks.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.action_log_probs.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.advantages.sizes().vec() == std::vector<long>{5, 1});
    }

    SUBCASE("done() indicates whether the generator has finished")
    {
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(generator.done());
    }

    SUBCASE("Calling a generator after it has finished throws an exception")
    {
        generator.next();
        generator.next();
        generator.next();
        CHECK_THROWS(generator.next());
    }
}
}