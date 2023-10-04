clang-format -style=file -i $(
    find src/ include/ test/ -type f \
    \( -name "*.h" -o -name "*.cpp" -o -name "*.cc" \
    -o -name "*.cu" -o -name "*.cuh" -o -name "*.py" \)
)
