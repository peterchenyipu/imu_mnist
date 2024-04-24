# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

set(BOARD "xiao_ble_sense")
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ble_hid_app)

include(utils/cmake/utils.cmake)

# Needed for colorful output
zephyr_compile_options(-fdiagnostics-color=always)

# Use hardware acceleration for DSP and Neural Network code
# You'll need to disable these on non-Arm cores
add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=1
                -DARM_MATH_LOOPUNROLL
                )

# Add the Edge Impulse SDK
add_subdirectory(edge-impulse-sdk/cmake/zephyr)

# Edge impulse SDK include directories
set(INCLUDES
    .
    src
    tflite-model
    model-parameters
    include
    )

include_directories(${INCLUDES})


RECURSIVE_FIND_FILE(MODEL_FILES "tflite-model" "*.cpp")
list(APPEND SOURCE_FILES ${MODEL_FILES})

FILE(GLOB_RECURSE app_sources src/*.c)
list(APPEND SOURCE_FILES ${app_sources})

FILE(GLOB_RECURSE app_sources src/*.cpp)
list(APPEND SOURCE_FILES ${app_sources})

target_sources(app PRIVATE
  ${SOURCE_FILES}
  )
# target_sources(app PRIVATE src/main.cpp)


