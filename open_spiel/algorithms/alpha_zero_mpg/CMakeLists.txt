# To enable C++ AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC. See:
# https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md
if (OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  add_library (alpha_zero_parametric OBJECT
          parametric_alpha_zero.h
          parametric_alpha_zero.cc
    device_manager.h
          pvpevaluator.h
          pvpevaluator.cc
          pvpnet.h
          pvpnet.cc
  )

  set(TEST_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data/test/*
          ${CMAKE_CURRENT_SOURCE_DIR}/test_data/checkpoints/*
          ${CMAKE_CURRENT_SOURCE_DIR}/test_data/PVPNet_test_gnn.pb
          ${CMAKE_CURRENT_SOURCE_DIR}/test_data/PVPNet_test_mlp.pb
          )

  target_include_directories (alpha_zero_parametric PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pvpnet_test pvpnet_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:alpha_zero_parametric> $<TARGET_OBJECTS:tests>)
  add_custom_command(TARGET pvpnet_test POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${CMAKE_CURRENT_BINARY_DIR}/test_data
          DEPENDS ${TEST_FILES})
  add_test(pvpnet_test pvpnet_test)
  add_executable(pvpnet_inference_test pvpnet_inference_test.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:alpha_zero_parametric> $<TARGET_OBJECTS:tests>)
  add_test(pvpnet_inference_test pvpnet_inference_test)

  target_link_libraries(alpha_zero_parametric TensorflowCC::TensorflowCC)
  target_link_libraries(pvpnet_test TensorflowCC::TensorflowCC)
  target_link_libraries(pvpnet_inference_test TensorflowCC::TensorflowCC)

endif()
