#include <memory>

#include <torch/torch.h>

#include "cpprl/model/output_layers.h"
#include "cpprl/model/model_utils.h"
#include "cpprl/distributions/distribution.h"
#include "cpprl/distributions/categorical.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
CategoricalOutput::CategoricalOutput(unsigned int num_inputs,
                                     unsigned int num_outputs)
    : linear(num_inputs, num_outputs)
{
    register_module("linear", linear);
    init_weights(linear->named_parameters(), 0.01, 0);
}

std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor x)
{
    auto y = x;
    x = linear(x);
    return std::make_unique<Categorical>(nullptr, &x);
}

TEST_CASE("CategoricalOutput")
{
    auto output_layer = CategoricalOutput(3, 5);

    SUBCASE("Output distribution has correct output shape")
    {
        float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
        auto input_tensor = torch::from_blob(input_array,
                                             {2, 3},
                                             TensorOptions(torch::kFloat));
        auto dist = output_layer.forward(input_tensor);

        auto output = dist->sample();

        CHECK(output.sizes().vec() == std::vector<long>{2});
    }
}
}