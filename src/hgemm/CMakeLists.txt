add_myblas_sublibrary(hgemm)
target_compile_options(hgemm PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)