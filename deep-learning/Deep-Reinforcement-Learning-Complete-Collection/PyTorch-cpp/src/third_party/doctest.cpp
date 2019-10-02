#define DOCTEST_CONFIG_IMPLEMENT

#include <spdlog/spdlog.h>

#include "third_party/doctest.h"

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::off);
    return doctest::Context(argc, argv).run();
}