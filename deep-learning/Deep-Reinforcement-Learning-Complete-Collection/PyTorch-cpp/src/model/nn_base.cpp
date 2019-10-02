#include <torch/torch.h>

#include "cpprl/model/nn_base.h"
#include "cpprl/model/model_utils.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
NNBase::NNBase(bool recurrent,
               unsigned int recurrent_input_size,
               unsigned int hidden_size)
    : recurrent(recurrent), hidden_size(hidden_size), gru(nullptr)
{
    // Init GRU
    if (recurrent)
    {
        gru = nn::GRU(nn::GRUOptions(recurrent_input_size, hidden_size));
        register_module("gru", gru);
        // Init weights
        init_weights(gru->named_parameters(), 1, 0);
    }
}

// Do not use.
//
// Instantiate a subclass and use their's instead
std::vector<torch::Tensor> NNBase::forward(torch::Tensor /*inputs*/,
                                           torch::Tensor /*hxs*/,
                                           torch::Tensor /*masks*/)
{
    return std::vector<torch::Tensor>();
}

unsigned int NNBase::get_hidden_size() const
{
    if (recurrent)
    {
        return hidden_size;
    }
    return 1;
}

std::vector<torch::Tensor> NNBase::forward_gru(torch::Tensor x,
                                               torch::Tensor rnn_hxs,
                                               torch::Tensor masks)
{
    if (x.size(0) == rnn_hxs.size(0))
    {
        auto gru_output = gru->forward(x.unsqueeze(0),
                                       (rnn_hxs * masks).unsqueeze(0));
        return {gru_output.output.squeeze(0), gru_output.state.squeeze(0)};
    }
    else
    {
        // x is a (timesteps, agents, -1) tensor that has been flattened to
        // (timesteps * agents, -1)
        auto agents = rnn_hxs.size(0);
        auto timesteps = x.size(0) / agents;

        // Unflatten
        x = x.view({timesteps, agents, x.size(1)});

        // Same for masks
        masks = masks.view({timesteps, agents});

        // Figure out which steps in the sequence have a zero for any agent
        // We assume the first timestep has a zero in it
        auto has_zeros = (masks.narrow(0, 1, masks.size(0) - 1) == 0)
                             .any(-1)
                             .nonzero()
                             .squeeze();

        // +1 to correct the masks[1:]
        has_zeros += 1;

        // Add t=0 and t=timesteps to the list
        // has_zeros = [0] + has_zeros + [timesteps]
        has_zeros = has_zeros.contiguous().to(torch::kInt);
        std::vector<int> has_zeros_vec(
            has_zeros.data<int>(),
            has_zeros.data<int>() + has_zeros.numel());
        has_zeros_vec.insert(has_zeros_vec.begin(), {0});
        has_zeros_vec.push_back(timesteps);

        rnn_hxs = rnn_hxs.unsqueeze(0);
        std::vector<torch::Tensor> outputs;
        for (unsigned int i = 0; i < has_zeros_vec.size() - 1; ++i)
        {
            // We can now process long runs of timesteps without dones in them in
            // one go
            auto start_idx = has_zeros_vec[i];
            auto end_idx = has_zeros_vec[i + 1];

            auto gru_output = gru(
                x.index({torch::arange(start_idx,
                                       end_idx,
                                       TensorOptions(ScalarType::Long))}),
                rnn_hxs * masks[start_idx].view({1, -1, 1}));

            outputs.push_back(gru_output.output);
        }

        // x is a (timesteps, agents, -1) tensor
        x = torch::cat(outputs, 0).squeeze(0);
        x = x.view({timesteps * agents, -1});
        rnn_hxs = rnn_hxs.squeeze(0);

        return {x, rnn_hxs};
    }
}

TEST_CASE("NNBase")
{
    auto base = std::make_shared<NNBase>(true, 5, 10);

    SUBCASE("forward_gru() outputs correct shapes when given samples from one"
            " agent")
    {
        auto inputs = torch::rand({4, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }

    SUBCASE("forward_gru() outputs correct shapes when given samples from "
            "multiple agents")
    {
        auto inputs = torch::rand({12, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({12, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 12);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }
}
}