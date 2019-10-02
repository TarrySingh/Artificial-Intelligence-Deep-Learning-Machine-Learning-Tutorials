#pragma once

#include <string>
#include <vector>

#include "cpprl/storage.h"

namespace cpprl
{
struct UpdateDatum
{
    std::string name;
    float value;
};

class Algorithm
{
  public:
    virtual ~Algorithm() = 0;

    virtual std::vector<UpdateDatum> update(RolloutStorage &rollouts) = 0;
};

inline Algorithm::~Algorithm() {}
}