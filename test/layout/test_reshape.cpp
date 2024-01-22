#include <gtest/gtest.h>

#include "myblas_internal/layout.h"

using namespace myblas;

#define PRINT_RESHAPED_INDEX(tid, index)                     \
  printf("tid: %d\n", tid);                                  \
  printf("index: \n");                                       \
  print_index(index);                                        \
  printf("linear: %d\n", get_linear_idx<StrideType>(index)); \
  printf("--------------------\n");

TEST(TestReshape, shape_0) {
  constexpr int thread_local_m = 8;
  constexpr int thread_local_n = 4;

  constexpr int total_threads = thread_local_m * thread_local_n;

  using ShapeType = Shape<Int<thread_local_m>, Int<thread_local_n>>;
  using StrideType = Stride<Int<4>, Int<1>>;

  for (int tid = 0; tid < total_threads; ++tid) {
    auto index = reshape<ShapeType, StrideType>(tid);

    EXPECT_TRUE(get_linear_idx<StrideType>(index) == tid);
    //    PRINT_RESHAPED_INDEX(tid, index);
  }
}

TEST(TestReshape, shape_1) {
  constexpr int thread_local_m = 8;
  constexpr int total_threads = thread_local_m;

  using ShapeType = Shape<Int<thread_local_m>>;
  using StrideType = Stride<Int<1>>;

  for (int tid = 0; tid < total_threads; ++tid) {
    auto index = reshape<ShapeType, StrideType>(tid);

    EXPECT_TRUE(get_linear_idx<StrideType>(index) == tid);
    //    PRINT_RESHAPED_INDEX(tid, index);
  }
}

TEST(TestReshape, shape_2) {
  constexpr int thread_local_m = 8;
  constexpr int thread_local_n = 2;

  constexpr int total_threads = thread_local_m * thread_local_n;

  using ShapeType = Shape<Int<thread_local_m>, Int<thread_local_n>>;
  using StrideType = Stride<Int<1>, Int<8>>;

  for (int tid = 0; tid < total_threads; ++tid) {
    auto index = reshape<ShapeType, StrideType>(tid);

    EXPECT_TRUE(get_linear_idx<StrideType>(index) == tid);
    //    PRINT_RESHAPED_INDEX(tid, index);
  }
}

TEST(TestReshape, shape_3) {
  constexpr int warp_size = 32;
  constexpr int thread_group_m = 2;
  constexpr int thread_group_n = 2;
  constexpr int thread_local_m = 8;
  constexpr int thread_local_n = 1;

  constexpr int total_threads = warp_size;

  using ShapeType = Shape<Shape<Int<thread_local_m>, Int<thread_local_n>>,
                          Shape<Int<thread_group_m>, Int<thread_group_n>>>;
  using StrideType = Stride<Stride<Int<1>, Int<0>>, Stride<Int<8>, Int<16>>>;

  for (int tid = 0; tid < total_threads; ++tid) {
    auto index = reshape<ShapeType, StrideType>(tid);

    EXPECT_TRUE(get_linear_idx<StrideType>(index) == tid);
    //    PRINT_RESHAPED_INDEX(tid, index);
  }
}

TEST(TestReshape, shape_4) {
  constexpr int warp_m = 2;
  constexpr int warp_n = 4;
  constexpr int warp_per_block = warp_m * warp_n;
  constexpr int warp_size = 32;
  constexpr int thread_group_m = 2;
  constexpr int thread_group_n = 2;
  constexpr int thread_local_m = 8;
  constexpr int thread_local_n = 1;

  constexpr int total_threads = warp_size * warp_per_block;

  using ShapeType = Shape<Shape<Int<thread_local_m>, Int<thread_local_n>>,
                          Shape<Int<thread_group_m>, Int<thread_group_n>>,
                          Shape<Int<warp_m>, Int<warp_n>>>;
  using StrideType = Stride<Stride<Int<1>, Int<0>>, Stride<Int<8>, Int<16>>,
                            Stride<Int<128>, Int<32>>>;

  for (int tid = 0; tid < total_threads; ++tid) {
    auto index = reshape<ShapeType, StrideType>(tid);

    EXPECT_TRUE(get_linear_idx<StrideType>(index) == tid);
    //    PRINT_RESHAPED_INDEX(tid, index);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
