target_sources(cpprl
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cnn_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlp_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nn_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/output_layers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/policy.cpp
)

if (CPPRL_BUILD_TESTS)
    target_sources(cpprl_tests
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cnn_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/mlp_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/model_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/nn_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/output_layers.cpp
        ${CMAKE_CURRENT_LIST_DIR}/policy.cpp
    )
endif (CPPRL_BUILD_TESTS)