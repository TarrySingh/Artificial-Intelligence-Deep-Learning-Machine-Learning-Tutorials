#include <torch/torch.h>

#include "cpprl/model/cnn_base.h"
#include "cpprl/model/model_utils.h"
#include "third_party/doctest.h"

namespace cpprl
{
CnnBase::CnnBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, hidden_size, hidden_size),
      main(nn::Conv2d(nn::Conv2dOptions(num_inputs, 32, 8).stride(4)),
           nn::Functional(torch::relu),
           nn::Conv2d(nn::Conv2dOptions(32, 64, 4).stride(2)),
           nn::Functional(torch::relu),
           nn::Conv2d(nn::Conv2dOptions(64, 32, 3).stride(1)),
           nn::Functional(torch::relu),
           Flatten(),
           nn::Linear(32 * 7 * 7, hidden_size),
           nn::Functional(torch::relu)),
      critic_linear(nn::Linear(hidden_size, 1))
{
    register_module("main", main);
    register_module("critic_linear", critic_linear);

    init_weights(main->named_parameters(), sqrt(2.), 0);
    init_weights(critic_linear->named_parameters(), 1, 0);

    train();
}

std::vector<torch::Tensor> CnnBase::forward(torch::Tensor inputs,
                                            torch::Tensor rnn_hxs,
                                            torch::Tensor masks)
{
    auto x = main->forward(inputs / 255.);

    if (is_recurrent())
    {
        auto gru_output = forward_gru(x, rnn_hxs, masks);
        x = gru_output[0];
        rnn_hxs = gru_output[1];
    }

    return {critic_linear->forward(x), x, rnn_hxs};
}

TEST_CASE("CnnBase")
{
    auto base = std::make_shared<CnnBase>(3, true, 10);

    SUBCASE("Sanity checks")
    {
        CHECK(base->is_recurrent() == true);
        CHECK(base->get_hidden_size() == 10);
    }

    SUBCASE("Output tensors are correct shapes")
    {
        auto inputs = torch::rand({4, 3, 84, 84});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward(inputs, rnn_hxs, masks);

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