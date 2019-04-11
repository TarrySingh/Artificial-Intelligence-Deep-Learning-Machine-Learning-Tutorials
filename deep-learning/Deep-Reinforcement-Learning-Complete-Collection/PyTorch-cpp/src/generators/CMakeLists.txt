target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/feed_forward_generator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/recurrent_generator.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/feed_forward_generator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/recurrent_generator.cpp
    )
endif (CPPRL_BUILD_TESTS)