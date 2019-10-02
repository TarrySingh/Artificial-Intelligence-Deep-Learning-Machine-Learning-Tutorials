add_executable(gym_client gym_client.cpp communicator.cpp)

target_include_directories(gym_client 
    PRIVATE
    .
    ../include
    ../lib/msgpack-c/include
    ../lib/spdlog/include
)

target_link_libraries(gym_client PRIVATE zmq cpprl)