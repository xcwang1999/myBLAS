#include <gtest/gtest.h>
#include "myblas_internal/layout.h"

namespace {
  using namespace myblas;
  constexpr int a1 = 1, b1 = 2, c1 = 3, d1 = 4, e1 = 5, f1 = 6;
  int a2 = 11, b2 = 12, c2 = 13, d2 = 14, e2 = 15, f2 = 16;
}

TEST(TestLinear, shape_0){
  auto stride = make_stride(make_stride(Int<a1>{}, Int<b1>{}), make_stride(Int<c1>{}, Int<d1>{}));
  auto index = make_index(make_index(DynamicInt{a2}, DynamicInt{b2}), make_index(DynamicInt{c2}, DynamicInt{d2}));

  int linear = get_linear_idx<decltype(stride)>(index);

  EXPECT_TRUE(linear == a1*a2 + b1*b2 + c1*c2 + d1*d2);
}

TEST(TestLinear, shape_1){
  auto stride = make_stride(Int<a1>{}, make_stride(Int<c1>{}, Int<d1>{}));
  auto index = make_index(DynamicInt{a2}, make_index(DynamicInt{c2}, DynamicInt{d2}));

  int linear = get_linear_idx<decltype(stride)>(index);

  EXPECT_TRUE(linear == a1*a2 + c1*c2 + d1*d2);
}

TEST(TestLinear, shape_2){
  auto stride = make_stride(Int<a1>{}, Int<c1>{});
  auto index = make_index(DynamicInt{a2}, DynamicInt{c2});

  int linear = get_linear_idx<decltype(stride)>(index);

  EXPECT_TRUE(linear == a1*a2 + c1*c2);
}

TEST(TestLinear, shape_3){
  auto stride = make_stride(
      make_stride(Int<a1>{}, Int<b1>{}, Int<c1>{}),
      make_stride(Int<d1>{}, Int<e1>{}, Int<f1>{})
  );
  auto index = make_index(
      make_index(DynamicInt{a2}, DynamicInt{b2}, DynamicInt{c2}),
      make_index(DynamicInt{d2}, DynamicInt{e2}, DynamicInt{f2})
  );

  int linear = get_linear_idx<decltype(stride)>(index);

  EXPECT_TRUE(linear == a1*a2 + b1*b2 + c1*c2 + d1*d2 + e1*e2 + f1*f2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
