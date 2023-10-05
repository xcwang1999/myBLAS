#! /bin/sh

git submodule update --init --recursive

cmake -S. -B build \
    -DBUILD_WITHOUT_LAPACK=ON \