cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

project(Bairstow LANGUAGES CXX)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

include(../specific.cmake)
# cpmaddpackage( NAME fmt GIT_TAG 8.1.1 GITHUB_REPOSITORY fmtlib/fmt OPTIONS "FMT_INSTALL YES" #
# create an installable target )
#
# set(THREADS_PREFER_PTHREAD_FLAG ON) find_package(Threads REQUIRED)
#
# cpmaddpackage("gh:microsoft/GSL@3.1.0")
#
find_package(${PROJECT_NAME} REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../standalone ${CMAKE_BINARY_DIR}/test_installed)
