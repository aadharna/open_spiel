add_library (tests OBJECT
  basic_tests.h
  basic_tests.cc
  console_play_test.h
  console_play_test.cc
)
target_include_directories (tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(spiel_test spiel_test.cc
               $<TARGET_OBJECTS:tests> ${OPEN_SPIEL_OBJECTS})
add_test(spiel_test spiel_test)

add_executable(action_view_test action_view_test.cc ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(action_view_test action_view_test)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;)

add_executable(python_interpreter_test python_interpreter_test.cpp ../interpreter/interpreter_adapter.cc
        ../interpreter/interpreter_adapter.h ../reverb/reverb_interpreter.h ../reverb/reverb_interpreter.cpp
        ../interpreter/representation.cpp ../interpreter/representation.h ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
include_directories (../pybind11_abseil ../../pybind11/include)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} "../cmake")
find_package(pybind11 REQUIRED)
target_link_libraries(python_interpreter_test PRIVATE pybind11::embed )
add_test(python_interpreter_test python_interpreter_test)

if (BUILD_SHARED_LIB)
  if (WIN32)
    add_executable(shared_lib_test shared_lib_test.cc ${OPEN_SPIEL_OBJECTS})
  else()
    add_executable(shared_lib_test shared_lib_test.cc)
  endif()
  target_link_libraries(shared_lib_test open_spiel)
  add_test(shared_lib_test shared_lib_test)
endif()
