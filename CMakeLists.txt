# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

set(BOARD "xiao_ble_sense")
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ble_hid_app)

add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=1
                -DARM_MATH_LOOPUNROLL
                )

target_include_directories(app PRIVATE include)
FILE(GLOB_RECURSE app_sources src/*.c)
target_sources(app PRIVATE
  ${app_sources}
  )
target_sources(app PRIVATE src/main.cpp)


