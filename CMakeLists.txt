CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
SET(CMAKE_C_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)

PROJECT(Gdev)
SET(serial "1.0.0")
SET(soserial "1")

## Option
OPTION(driver "select GPU driver. must set driver name in the user-space mode" nouveau)
OPTION(user "user mode(default off)" OFF)
OPTION(runtime "enable CUDA runtime API(default on)" ON)
OPTION(usched "enable user mode scheduler(default off)" OFF)
OPTION(use_as "use assembler(default off)" OFF)
OPTION(use_llvm "use LLVM compiler(default off)" OFF)
OPTION(use_open64 "use Open64(nvopencc) compiler(default off)" OFF)

MACRO(INCLUDE_DIRECTORY_IF_EXISTS DIR)
    IF(EXISTS "${DIR}/")
        INCLUDE_DIRECTORIES(${DIR})
    ENDIF()
ENDMACRO()

MACRO(LINK_DIRECTORY_IF_EXISTS DIR)
    if(EXISTS "${DIR}/")
        LINK_DIRECTORIES(${DIR})
    endif()
ENDMACRO()

include(ExternalProject)
ExternalProject_Add(
    envytools
    URL https://github.com/envytools/envytools/archive/64a2605897a3ad6b37e587337bfb05e770129e37.tar.gz
    # DOWNLOAD_NAME envytools.tar.gz
    # GIT_REPOSITORY https://github.com/envytools/envytools.git
    # GIT_TAG 64a2605897a3ad6b37e587337bfb05e770129e37
    TIMEOUT 10
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND cmake -H<SOURCE_DIR>/. -B<BINARY_DIR> -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_C_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BUILD_COMMAND make -C <BINARY_DIR>
    INSTALL_COMMAND make  -C <BINARY_DIR> install
    INSTALL_DIR ${PROJECT_BINARY_DIR}
    LOG_DOWNLOAD OFF
    LOG_CONFIGURE OFF
    LOG_BUILD OFF)
ADD_LIBRARY(nva STATIC IMPORTED)
SET_TARGET_PROPERTIES(nva PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libnva.a)
ADD_DEPENDENCIES(nva envytools)
ADD_LIBRARY(nvhw STATIC IMPORTED)
SET_TARGET_PROPERTIES(nvhw PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libnvhw.a)
ADD_DEPENDENCIES(nvhw envytools)
ADD_LIBRARY(envy STATIC IMPORTED)
SET_TARGET_PROPERTIES(envy PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libenvy.a)
ADD_DEPENDENCIES(envy envytools)

## Generate gdev_autogen.h
STRING(TOUPPER "${driver}" DRIVER_NAME)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/common/gdev_autogen.h.in ${CMAKE_CURRENT_BINARY_DIR}/gdev_autogen.h)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

IF(user)
    INCLUDE_DIRECTORY_IF_EXISTS(/usr/local/include/libdrm)
    LINK_DIRECTORY_IF_EXISTS(/usr/local/lib)
ENDIF()

INCLUDE_DIRECTORIES(util)
INCLUDE_DIRECTORIES(common)

FIND_PACKAGE(Doxygen)
include(UseDoxygen OPTIONAL)

ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(lib)

IF(NOT user)
    ADD_SUBDIRECTORY(mod)
ENDIF(NOT user)

IF(use_as)
    ADD_SUBDIRECTORY(compiler/as)
ENDIF(use_as)

IF(use_llvm)
    ADD_SUBDIRECTORY(compiler/llvm)
ENDIF(use_llvm)

IF(use_open64)
    ADD_SUBDIRECTORY(compiler/open64)
ENDIF(use_open64)
