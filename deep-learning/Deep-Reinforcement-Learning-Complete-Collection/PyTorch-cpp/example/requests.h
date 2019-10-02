#pragma once

#include <string>

#include <msgpack.hpp>

namespace gym_client
{
template <class T>
struct Request
{
    Request(const std::string &method, std::shared_ptr<T> param) : method(method), param(param) {}

    std::string method;
    std::shared_ptr<T> param;
    MSGPACK_DEFINE_MAP(method, param)
};

struct InfoParam
{
    int x;
    MSGPACK_DEFINE_MAP(x);
};

struct MakeParam
{
    std::string env_name;
    float gamma;
    int num_envs;
    MSGPACK_DEFINE_MAP(env_name, gamma, num_envs);
};

struct ResetParam
{
    int x;
    MSGPACK_DEFINE_MAP(x);
};

struct StepParam
{
    std::vector<std::vector<int>> actions;
    bool render;
    MSGPACK_DEFINE_MAP(actions, render);
};

struct InfoResponse
{
    std::string action_space_type;
    std::vector<long> action_space_shape;
    std::string observation_space_type;
    std::vector<long> observation_space_shape;
    MSGPACK_DEFINE_MAP(action_space_type, action_space_shape,
                       observation_space_type, observation_space_shape);
};

struct MakeResponse
{
    std::string result;
    MSGPACK_DEFINE_MAP(result);
};

struct CnnResetResponse
{
    std::vector<std::vector<std::vector<std::vector<float>>>> observation;
    MSGPACK_DEFINE_MAP(observation);
};

struct MlpResetResponse
{
    std::vector<std::vector<float>> observation;
    MSGPACK_DEFINE_MAP(observation);
};

struct StepResponse
{
    std::vector<std::vector<float>> reward;
    std::vector<std::vector<bool>> done;
    std::vector<std::vector<float>> real_reward;
};

struct CnnStepResponse : StepResponse
{
    std::vector<std::vector<std::vector<std::vector<float>>>> observation;
    MSGPACK_DEFINE_MAP(observation, reward, done, real_reward);
};

struct MlpStepResponse : StepResponse
{
    std::vector<std::vector<float>> observation;
    MSGPACK_DEFINE_MAP(observation, reward, done, real_reward);
};
}
