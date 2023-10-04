#include <iostream>
#include "matrix/matrixGPU.cuh"
#include "test_hgemm_padding.cuh"
#include "test_sgemm.cuh"
#include "test_sgemm_padding.cuh"
#include "test_wrapper_class.cuh"
int main() {
  int shapes[] = {1111};
  int repeat = 1;
  // const int num = 16;
  // int shapes[num];
  // for(int n=1; n<=num; n++)
  //     shapes[n-1] = n * 1024;

  testWrapper<float, MatrixGPUPitched<float>>(shapes, repeat);
  testSgemm(shapes, repeat);
  testSgemmPadding(shapes, repeat);

  testHgemmPadding(shapes, repeat);
  return 0;
}