add_myblas_unit_test(test_ldmatrix test_ldmatrix.cu)
add_myblas_unit_test(test_mma test_mma.cu)
target_compile_options(test_ldmatrix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(test_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
