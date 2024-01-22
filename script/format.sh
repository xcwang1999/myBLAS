#!/bin/bash

# Function to check if a program is installed
is_installed() {
    command -v "$1" &> /dev/null
}

# Directories and file extensions to consider
C_DIRECTORIES="src/ include/ test/ example/"
PY_DIRECTORIES="bin/ test/ example/"

IFS=' ' read -ra C_DIR_ARR <<< "$C_DIRECTORIES"
IFS=' ' read -ra PY_DIR_ARR <<< "$PY_DIRECTORIES"

# Format C/C++ files using clang-format if installed
if is_installed clang-format; then
    C_FILES=$(find $C_DIRECTORIES -type f \
    \( -name "*.h" -o \
       -name "*.hpp" -o \
       -name "*.cuh" -o \
       -name "*.c" -o \
       -name "*.cc" -o \
       -name "*.cpp" -o \
       -name "*.cxx" -o \
       -name "*.cu" \))
    if [[ ! -z "$C_FILES" ]]; then
        echo "$C_FILES" | xargs clang-format -style=file -i
        echo "Formatted C/C++ files in following directories:"
        for dir in "${C_DIR_ARR[@]}"; do
            echo "$dir"
        done
    else
        echo "No C/C++ files found for formatting."
    fi
else
    echo "clang-format is not installed. Skipping C/C++ file formatting."
fi

# Format Python files using black if installed
if is_installed black; then
    PY_FILES=$(find $PY_DIRECTORIES -type f -name "*.py")
    if [[ ! -z "$PY_FILES" ]]; then
        echo "$PY_FILES" | xargs black
        for dir in "${PY_DIR_ARR[@]}"; do
            echo "$dir"
        done
    else
        echo "No Python files found for formatting."
    fi
else
    echo "black is not installed. Skipping Python file formatting."
fi