#include <algorithm>
#include <vector>

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "cpprl/generators/feed_forward_generator.h"
#include "cpprl/generators/generator.h"
#include "third_party/doctest.h"

namespace cpprl
{
FeedForwardGenerator::FeedForwardGenerator(int mini_batch_size,
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
      index(0)
{
    int batch_size = advantages.numel();
    indices = torch::randperm(batch_size,
                              torch::TensorOptions(torch::kLong))
                  .view({-1, mini_batch_size});
}

bool FeedForwardGenerator::done() const
{
    return index >= indices.size(0);
}

MiniBatch FeedForwardGenerator::next()
{
    if (index >= indices.size(0))
    {
        spdlog::error("No minibatches left in generator. Index {}, minibatch "
                      "count: {}.",
                      index, indices.size(0));
        throw std::exception();
    }

    MiniBatch mini_batch;

    int timesteps = observations.size(0) - 1;

    auto observations_shape = observations.sizes().vec();
    observations_shape.erase(observations_shape.begin());
    observations_shape[0] = -1;
    mini_batch.observations = observations.narrow(0, 0, timesteps)
                                  .view(observations_shape)
                                  .index(indices[index]);
    mini_batch.hidden_states = hidden_states.narrow(0, 0, timesteps)
                                   .view({-1, hidden_states.size(-1)})
                                   .index(indices[index]);
    mini_batch.actions = actions.view({-1, actions.size(-1)})
                             .index(indices[index]);
    mini_batch.value_predictions = value_predictions.narrow(0, 0, timesteps)
                                       .view({-1, 1})
                                       .index(indices[index]);
    mini_batch.returns = returns.narrow(0, 0, timesteps)
                             .view({-1, 1})
                             .index(indices[index]);
    mini_batch.masks = masks.narrow(0, 0, timesteps)
                           .view({-1, 1})
                           .index(indices[index]);
    mini_batch.action_log_probs = action_log_probs.view({-1, 1})
                                      .index(indices[index]);
    mini_batch.advantages = advantages.view({-1, 1})
                                .index(indices[index]);

    index++;
    return mini_batch;
}

TEST_CASE("FeedForwardGenerator")
{
    FeedForwardGenerator generator(5, torch::rand({6, 3, 4}), torch::rand({6, 3, 3}),
                                   torch::rand({5, 3, 1}), torch::rand({6, 3, 1}),
                                   torch::rand({6, 3, 1}), torch::ones({6, 3, 1}),
                                   torch::rand({5, 3, 1}), torch::rand({5, 3, 1}));

    SUBCASE("Minibatch tensors are correct sizes")
    {
        auto minibatch = generator.next();

        CHECK(minibatch.observations.sizes().vec() == std::vector<long>{5, 4});
        CHECK(minibatch.hidden_states.sizes().vec() == std::vector<long>{5, 3});
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