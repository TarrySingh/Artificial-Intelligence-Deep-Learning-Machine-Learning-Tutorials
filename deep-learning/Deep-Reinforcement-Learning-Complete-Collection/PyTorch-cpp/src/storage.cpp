#include <memory>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "cpprl/generators/feed_forward_generator.h"
#include "cpprl/generators/recurrent_generator.h"
#include "cpprl/storage.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

namespace cpprl
{
RolloutStorage::RolloutStorage(unsigned int num_steps,
                               unsigned int num_processes,
                               c10::ArrayRef<int64_t> obs_shape,
                               ActionSpace action_space,
                               unsigned int hidden_state_size,
                               torch::Device device)
    : device(device), num_steps(num_steps), step(0)
{
    std::vector<long> observations_shape{num_steps + 1, num_processes};
    observations_shape.insert(observations_shape.end(), obs_shape.begin(),
                              obs_shape.end());
    observations = torch::zeros(observations_shape, torch::TensorOptions(device));
    hidden_states = torch::zeros({num_steps + 1, num_processes,
                                  hidden_state_size},
                                 torch::TensorOptions(device));
    rewards = torch::zeros({num_steps, num_processes, 1}, torch::TensorOptions(device));
    value_predictions = torch::zeros({num_steps + 1, num_processes, 1}, torch::TensorOptions(device));
    returns = torch::zeros({num_steps + 1, num_processes, 1}, torch::TensorOptions(device));
    action_log_probs = torch::zeros({num_steps, num_processes, 1}, torch::TensorOptions(device));
    int num_actions;
    if (action_space.type == "Discrete")
    {
        num_actions = 1;
    }
    else
    {
        num_actions = action_space.shape[0];
    }
    actions = torch::zeros({num_steps, num_processes, num_actions}, torch::TensorOptions(device));
    if (action_space.type == "Discrete")
    {
        actions = actions.to(torch::kLong);
    }
    masks = torch::ones({num_steps + 1, num_processes, 1}, torch::TensorOptions(device));
}

void RolloutStorage::after_update()
{
    observations[0].copy_(observations[-1]);
    hidden_states[0].copy_(hidden_states[-1]);
    masks[0].copy_(masks[-1]);
}

void RolloutStorage::compute_returns(torch::Tensor next_value,
                                     bool use_gae,
                                     float gamma,
                                     float tau)
{
    if (use_gae)
    {
        value_predictions[-1] = next_value;
        torch::Tensor gae = torch::zeros({rewards.size(1), 1}, torch::TensorOptions(device));
        for (int step = rewards.size(0) - 1; step >= 0; --step)
        {
            auto delta = (rewards[step] +
                          gamma *
                              value_predictions[step + 1] *
                              masks[step + 1] -
                          value_predictions[step]);
            gae = delta + gamma * tau * masks[step + 1] * gae;
            returns[step] = gae + value_predictions[step];
        }
    }
    else
    {
        returns[-1] = next_value;
        for (int step = rewards.size(0) - 1; step >= 0; --step)
        {
            returns[step] = (returns[step + 1] *
                                 gamma *
                                 masks[step + 1] +
                             rewards[step]);
        }
    }
}

std::unique_ptr<Generator> RolloutStorage::feed_forward_generator(
    torch::Tensor advantages, int num_mini_batch)
{
    auto num_steps = actions.size(0);
    auto num_processes = actions.size(1);
    auto batch_size = num_processes * num_steps;
    if (batch_size < num_mini_batch)
    {
        spdlog::error("PPO needs the number of processes ({}) * the number of "
                      "steps ({}) = {} to be greater than or equal to the number "
                      "of minibatches ({})",
                      num_processes,
                      num_steps,
                      num_mini_batch);
        throw std::exception();
    }
    auto mini_batch_size = batch_size / num_mini_batch;
    return std::make_unique<FeedForwardGenerator>(
        mini_batch_size,
        observations,
        hidden_states,
        actions,
        value_predictions,
        returns,
        masks,
        action_log_probs,
        advantages);
}

void RolloutStorage::insert(torch::Tensor observation,
                            torch::Tensor hidden_state,
                            torch::Tensor action,
                            torch::Tensor action_log_prob,
                            torch::Tensor value_prediction,
                            torch::Tensor reward,
                            torch::Tensor mask)
{
    observations[step + 1].copy_(observation);
    hidden_states[step + 1].copy_(hidden_state);
    actions[step].copy_(action);
    action_log_probs[step].copy_(action_log_prob);
    value_predictions[step].copy_(value_prediction);
    rewards[step].copy_(reward);
    masks[step + 1].copy_(mask);

    step = (step + 1) % num_steps;
}

std::unique_ptr<Generator> RolloutStorage::recurrent_generator(
    torch::Tensor advantages, int num_mini_batch)
{
    auto num_processes = actions.size(1);
    if (num_processes < num_mini_batch)
    {
        spdlog::error("PPO needs the number of processes ({}) to be greater than or"
                      " equal to the number of minibatches ({})",
                      num_processes,
                      num_mini_batch);
        throw std::exception();
    }
    return std::make_unique<RecurrentGenerator>(
        num_processes,
        num_mini_batch,
        observations,
        hidden_states,
        actions,
        value_predictions,
        returns,
        masks,
        action_log_probs,
        advantages);
}

void RolloutStorage::set_first_observation(torch::Tensor observation)
{
    observations[0].copy_(observation);
}

void RolloutStorage::to(torch::Device device)
{
    this->device = device;
    observations = observations.to(device);
    hidden_states = hidden_states.to(device);
    rewards = rewards.to(device);
    value_predictions = value_predictions.to(device);
    returns = returns.to(device);
    action_log_probs = action_log_probs.to(device);
    actions = actions.to(device);
    masks = masks.to(device);
}

// cppcheck-suppress syntaxError
TEST_CASE("RolloutStorage")
{
    SUBCASE("Initializes tensors to correct sizes")
    {
        RolloutStorage storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);

        CHECK(storage.get_observations().size(0) == 4);
        CHECK(storage.get_observations().size(1) == 5);
        CHECK(storage.get_observations().size(2) == 5);
        CHECK(storage.get_observations().size(3) == 2);

        CHECK(storage.get_hidden_states().size(0) == 4);
        CHECK(storage.get_hidden_states().size(1) == 5);
        CHECK(storage.get_hidden_states().size(2) == 10);

        CHECK(storage.get_rewards().size(0) == 3);
        CHECK(storage.get_rewards().size(1) == 5);
        CHECK(storage.get_rewards().size(2) == 1);

        CHECK(storage.get_value_predictions().size(0) == 4);
        CHECK(storage.get_value_predictions().size(1) == 5);
        CHECK(storage.get_value_predictions().size(2) == 1);

        CHECK(storage.get_returns().size(0) == 4);
        CHECK(storage.get_returns().size(1) == 5);
        CHECK(storage.get_returns().size(2) == 1);

        CHECK(storage.get_action_log_probs().size(0) == 3);
        CHECK(storage.get_action_log_probs().size(1) == 5);
        CHECK(storage.get_action_log_probs().size(2) == 1);

        CHECK(storage.get_actions().size(0) == 3);
        CHECK(storage.get_actions().size(1) == 5);
        CHECK(storage.get_actions().size(2) == 1);

        CHECK(storage.get_masks().size(0) == 4);
        CHECK(storage.get_masks().size(1) == 5);
        CHECK(storage.get_masks().size(2) == 1);
    }

    SUBCASE("Initializes actions to correct type")
    {
        SUBCASE("Long")
        {
            RolloutStorage storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);

            CHECK(storage.get_actions().dtype() == torch::kLong);
        }

        SUBCASE("Float")
        {
            RolloutStorage storage(3, 5, {5, 2}, ActionSpace{"Box", {3}}, 10, torch::kCPU);

            CHECK(storage.get_actions().dtype() == torch::kFloat);
        }
    }

    SUBCASE("to() doesn't crash")
    {
        RolloutStorage storage(3, 4, {5}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        storage.to(torch::kCPU);
    }

    SUBCASE("insert() inserts values")
    {
        RolloutStorage storage(3, 4, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        storage.insert(torch::rand({4, 5, 2}) + 1,
                       torch::rand({4, 10}) + 1,
                       torch::randint(1, 3, {4, 1}),
                       torch::rand({4, 1}) + 1,
                       torch::rand({4, 1}) + 1,
                       torch::rand({4, 1}) + 1,
                       torch::zeros({4, 1}));

        INFO("Observations: \n"
             << storage.get_observations() << "\n");
        CHECK(storage.get_observations()[1][0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Hidden states: \n"
             << storage.get_hidden_states() << "\n");
        CHECK(storage.get_hidden_states()[1][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Actions: \n"
             << storage.get_actions() << "\n");
        CHECK(storage.get_actions()[0][0][0].item().toInt() != 0);
        INFO("Action log probs: \n"
             << storage.get_action_log_probs() << "\n");
        CHECK(storage.get_action_log_probs()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Value predictions: \n"
             << storage.get_value_predictions() << "\n");
        CHECK(storage.get_value_predictions()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Rewards: \n"
             << storage.get_rewards() << "\n");
        CHECK(storage.get_rewards()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        INFO("Masks: \n"
             << storage.get_masks() << "\n");
        CHECK(storage.get_masks()[1][0][0].item().toInt() != 1);
    }

    SUBCASE("compute_returns()")
    {
        RolloutStorage storage(3, 2, {4}, ActionSpace{"Discrete", {3}}, 5, torch::kCPU);

        std::vector<float> value_preds{0, 1};
        std::vector<float> rewards{0, 1};
        std::vector<float> masks{1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        value_preds = {1, 2};
        rewards = {1, 2};
        masks = {1, 0};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        value_preds = {2, 3};
        rewards = {2, 3};
        masks = {1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(value_preds.data(), {2, 1}),
                       torch::from_blob(rewards.data(), {2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));

        SUBCASE("Gives correct results without GAE")
        {
            std::vector<float> next_values{0, 1};
            storage.compute_returns(torch::from_blob(&next_values[0], {2, 1}),
                                    false, 0.6, 0.6);

            INFO("Masks: \n"
                 << storage.get_masks() << "\n");
            INFO("Rewards: \n"
                 << storage.get_rewards() << "\n");
            INFO("Returns: \n"
                 << storage.get_returns() << "\n");
            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.32));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(1));
        }

        SUBCASE("Gives correct results with GAE")
        {
            std::vector<float> next_values{0, 1};
            storage.compute_returns(torch::from_blob(&next_values[0], {2, 1}),
                                    true, 0.6, 0.6);

            INFO("Masks: \n"
                 << storage.get_masks() << "\n");
            INFO("Rewards: \n"
                 << storage.get_rewards() << "\n");
            INFO("Value predictions: \n"
                 << storage.get_value_predictions() << "\n");
            INFO("Returns: \n"
                 << storage.get_returns() << "\n");
            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.032));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(0));
        }
    }

    SUBCASE("after_update() copies last observation, hidden state and mask to "
            "the 0th timestep")
    {
        RolloutStorage storage(3, 2, {3}, ActionSpace{"Discrete", {3}}, 2, torch::kCPU);

        std::vector<float> obs{0, 1, 2, 1, 2, 3};
        std::vector<float> hidden_states{0, 1, 0, 1};
        std::vector<float> masks{0, 1};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        obs = {0, 1, 2, 1, 2, 3};
        hidden_states = {0, 1, 0, 1};
        masks = {0, 1};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        obs = {5, 6, 7, 7, 8, 9};
        hidden_states = {1, 2, 3, 4};
        masks = {0, 0};
        storage.insert(torch::from_blob(obs.data(), {2, 3}),
                       torch::from_blob(hidden_states.data(), {2, 2}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(masks.data(), {2, 1}));
        storage.after_update();

        INFO("Observations: \n"
             << storage.get_observations() << "\n");
        CHECK(storage.get_observations()[0][0][1].item().toDouble() ==
              doctest::Approx(6));
        INFO("Hidden_states: \n"
             << storage.get_hidden_states() << "\n");
        CHECK(storage.get_hidden_states()[0][0][1].item().toDouble() ==
              doctest::Approx(2));
        INFO("Masks: \n"
             << storage.get_masks() << "\n");
        CHECK(storage.get_masks()[0][0][0].item().toDouble() ==
              doctest::Approx(0));
    }

    SUBCASE("Can create feed-forward generator")
    {
        RolloutStorage storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        auto generator = storage.feed_forward_generator(torch::rand({3, 5, 1}), 5);
        generator->next();
    }

    SUBCASE("Can create recurrent generator")
    {
        RolloutStorage storage(3, 5, {5, 2}, ActionSpace{"Discrete", {3}}, 10, torch::kCPU);
        auto generator = storage.recurrent_generator(torch::rand({3, 5, 1}), 5);
        generator->next();
    }
}
}