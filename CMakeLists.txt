cmake_minimum_required(VERSION 3.30)
project(wilson_loop_QWZ)

SET(TARGET "wilson_loop_QWZ")
SET(CMAKE_CXX_STANDARD 26)
# EIGEN
SET(EIGEN "/opt/eigen/share/eigen3/cmake")
# MKL
SET(MKL_ARCH "intel64")
SET(MKL_LINK "dynamic")
SET(MKL_INTERFACE_FULL "intel_lp64")
SET(MKL_THREADING "intel_thread")
SET(MKL_MPI "intelmpi")
# GCC
SET(GCC_TOOLCHAIN "/opt/gcc-14/")


add_executable(wilson_loop_QWZ main.cpp)

# --------------------------------------------------------------- #
#                MKL SET-UP
# --------------------------------------------------------------- #
MESSAGE(NOTICE "Setting up MKL ...")
IF(DEFINED ENV{MKLROOT})
    MESSAGE(STATUS "MKLROOT is defined; Intel OneAPI environment variables are likely already configured. ")
    FIND_PACKAGE(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT} REQUIRED)
    MESSAGE(STATUS "Imported oneMKL targets : ${MKL_IMPORTED_TARGETS}")
ELSE ()
    MESSAGE(FATAL_ERROR "MKLROOT Environment variable is not yet defined. ")
ENDIF ()
# --------------------------------------------------------------- #


# --------------------------------------------------------------- #
#                Eigen Linear Algebra Headers
# --------------------------------------------------------------- #
MESSAGE(NOTICE "Setting up Eigen ...")
MESSAGE(NOTICE "Setting up Eigen: Will search in ${EIGEN}")
LIST(APPEND CMAKE_PREFIX_PATH ${EIGEN})

FIND_PACKAGE(Eigen3 3.3 REQUIRED NO_MODULE)
MESSAGE("Eigen3 Include Directories: ${EIGEN3_INCLUDE_DIRS}")
LINK_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
# ----------------------------------------------------------
##
##



# --------------------------------------------------------------- #
#                COMPILER OPTIONS
# --------------------------------------------------------------- #
MESSAGE(NOTICE "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
IF(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")

    MESSAGE(NOTICE "Added CXX Flags for use by compiler: ${CMAKE_CXX_COMPILER_ID}")
    TARGET_COMPILE_OPTIONS(${TARGET} PUBLIC
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS> "-O3;-Ofast;-xhost;-mtune=native;-openmp;-mavx;-Wall;-Wextra")

ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    TARGET_COMPILE_OPTIONS(${TARGET} PUBLIC
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS> "-O3;-march=native;-mtune=native;--gcc-toolchain=${GCC_TOOLCHAIN}")
    MESSAGE(NOTICE "Added CXX Flags for use by compiler: ${CMAKE_CXX_COMPILER_ID}")

ENDIF ()

# ---------------------------------------------------------------- #
TARGET_LINK_LIBRARIES(${TARGET} $<LINK_ONLY:MKL::MKL> Eigen3::Eigen)

TARGET_INCLUDE_DIRECTORIES(${TARGET} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

