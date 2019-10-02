#include <torch/torch.h>

#include "cpprl/model/mlp_base.h"
#include "cpprl/model/model_utils.h"
#include "third_party/doctest.h"

namespace cpprl
{
MlpBase::MlpBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, num_inputs, hidden_size),
      actor(nullptr),
      critic(nullptr),
      critic_linear(nullptr)
{
    if (recurrent)
    {
        num_inputs = hidden_size;
    }

    actor = nn::Sequential(nn::Linear(num_inputs, hidden_size),
                           nn::Functional(torch::tanh),
                           nn::Linear(hidden_size, hidden_size),
                           nn::Functional(torch::tanh));
    critic = nn::Sequential(nn::Linear(num_inputs, hidden_size),
                            nn::Functional(torch::tanh),
                            nn::Linear(hidden_size, hidden_size),
                            nn::Functional(torch::tanh));
    critic_linear = nn::Linear(hidden_size, 1);

    register_module("actor", actor);
    register_module("critic", critic);
    register_module("critic_linear", critic_linear);

    init_weights(actor->named_parameters(), sqrt(2.), 0);
    init_weights(critic->named_parameters(), sqrt(2.), 0);
    init_weights(critic_linear->named_parameters(), sqrt(2.), 0);

    train();
}

std::vector<torch::Tensor> MlpBase::forward(torch::Tensor inputs,
                                            torch::Tensor rnn_hxs,
                                            torch::Tensor masks)
{
    auto x = inputs;

    if (is_recurrent())
    {
        auto gru_output = forward_gru(x, rnn_hxs, masks);
        x = gru_output[0];
        rnn_hxs = gru_output[1];
    }

    auto hidden_critic = critic->forward(x);
    auto hidden_actor = actor->forward(x);

    return {critic_linear->forward(hidden_critic), hidden_actor, rnn_hxs};
}

TEST_CASE("MlpBase")
{
    SUBCASE("Recurrent")
    {
        auto base = MlpBase(5, true, 10);

        SUBCASE("Sanity checks")
        {
            CHECK(base.is_recurrent() == true);
            CHECK(base.get_hidden_size() == 10);
        }

        SUBCASE("Output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 5});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = base.forward(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 3);

            // Critic
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Actor
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 10);

            // Hidden state
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 10);
        }
    }

    SUBCASE("Non-recurrent")
    {
        auto base = MlpBase(5, false, 10);

        SUBCASE("Sanity checks")
        {
            CHECK(base.is_recurrent() == false);
        }

        SUBCASE("Output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 5});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = base.forward(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 3);

            // Critic
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Actor
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 10);

            // Hidden state
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 10);
        }
    }
}
}