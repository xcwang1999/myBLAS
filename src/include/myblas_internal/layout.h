#pragma once
#include <cstddef>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include "helper_macros.h"

namespace myblas {

template <int value_>
struct Int {
  static constexpr int value = value_;
};

struct DynamicInt {
  int value;
};

template <typename... Args>
struct ShapeBase {
  std::tuple<Args...> shape;

  MYBLAS_HOST_DEVICE constexpr ShapeBase() : shape() {}

  MYBLAS_HOST_DEVICE constexpr ShapeBase(Args... args)
      : shape(std::make_tuple(args...)) {}
};

template <typename... Args>
struct Shape : public ShapeBase<Args...> {
  using ShapeBase<Args...>::ShapeBase;
};

template <typename... Args>
struct Stride : public ShapeBase<Args...> {
  using ShapeBase<Args...>::ShapeBase;
};

template <typename... Args>
struct Index : public ShapeBase<Args...> {
  using ShapeBase<Args...>::ShapeBase;
};

template <typename... Args>
MYBLAS_HOST_DEVICE constexpr Shape<Args...> make_shape(Args... args) {
  return Shape<Args...>(args...);
}

template <typename... Args>
MYBLAS_HOST_DEVICE constexpr Stride<Args...> make_stride(Args... args) {
  return Stride<Args...>(args...);
}

template <typename... Args>
MYBLAS_HOST_DEVICE constexpr Index<Args...> make_index(Args... args) {
  return Index<Args...>(args...);
}

MYBLAS_HOST_DEVICE void print_dynamic_int(const DynamicInt& d) {
  printf("%d ", d.value);
}

template <typename T>
MYBLAS_HOST_DEVICE void print_element(const T& t) {
  if constexpr (std::is_same_v<T, DynamicInt>) {
    print_dynamic_int(t);
  } else {
    print_index(t);
  }
}

template <typename Tuple, std::size_t... Is>
MYBLAS_HOST_DEVICE void print_tuple_impl(const Tuple& t,
                                         std::index_sequence<Is...>) {
  (print_element(std::get<Is>(t)), ...);
}

template <typename... Args>
MYBLAS_HOST_DEVICE void print_index(const std::tuple<Args...>& t) {
  print_tuple_impl(
      t, std::make_index_sequence<std::tuple_size_v<std::tuple<Args...>>>{});
  printf("\n");
}

template <typename... Args>
MYBLAS_HOST_DEVICE void print_index(const Index<Args...>& s) {
  print_index(s.shape);
}

template <typename T>
struct to_dynamic_int_impl;

template <int N>
struct to_dynamic_int_impl<Int<N>> {
  MYBLAS_HOST_DEVICE static constexpr DynamicInt convert() {
    return DynamicInt{N};
  }
};

template <int... Ns>
struct to_dynamic_int_impl<Shape<Int<Ns>...>> {
  MYBLAS_HOST_DEVICE static constexpr auto convert() {
    return make_index(to_dynamic_int_impl<Int<Ns>>::convert()...);
  }
};

template <typename... Shapes>
struct to_dynamic_int_impl<Shape<Shapes...>> {
  MYBLAS_HOST_DEVICE static constexpr auto convert() {
    return make_index(to_dynamic_int_impl<Shapes>::convert()...);
  }
};

template <typename Shape>
MYBLAS_HOST_DEVICE constexpr auto to_dynamic_int() {
  return to_dynamic_int_impl<Shape>::convert();
}

template <typename Shape, typename Stride>
struct reshape_impl;

template <int... Ns, int... Ms>
struct reshape_impl<Shape<Int<Ns>...>, Stride<Int<Ms>...>> {
  MYBLAS_HOST_DEVICE static constexpr auto compute(const int tid) {
    return make_index(
        (Ms == 0 ? DynamicInt{0} : DynamicInt{(tid / Ms) % Ns})...);
  }
};

template <typename... ShapeTs, typename... StrideTs>
struct reshape_impl<Shape<ShapeTs...>, Stride<StrideTs...>> {
  MYBLAS_HOST_DEVICE static constexpr auto compute(const int tid) {
    return make_index(reshape_impl<ShapeTs, StrideTs>::compute(tid)...);
  }
};

template <int N, int M>
struct reshape_impl<Int<N>, Int<M>> {
  MYBLAS_HOST_DEVICE static constexpr DynamicInt compute(const int tid) {
    return M == 0 ? DynamicInt{0} : DynamicInt{(tid / M) % N};
  }
};

template <typename Shape, typename Stride>
MYBLAS_HOST_DEVICE constexpr auto reshape(const int tid) {
  return reshape_impl<Shape, Stride>::compute(tid);
}

template <typename Index, typename Stride>
struct dot_product_impl;

template <int N>
struct dot_product_impl<DynamicInt, Int<N>> {
  MYBLAS_HOST_DEVICE static constexpr int compute(const DynamicInt& idx,
                                                  const Int<N>&) {
    return N * idx.value;
  }
};

template <typename... IndexArgs, typename... StrideArgs>
struct dot_product_impl<Index<IndexArgs...>, Stride<StrideArgs...>> {
  MYBLAS_HOST_DEVICE static constexpr int compute(
      const Index<IndexArgs...>& idx, const Stride<StrideArgs...>& stride) {
    return compute_impl(idx, stride, std::index_sequence_for<IndexArgs...>{});
  }

 private:
  template <std::size_t... Is>
  MYBLAS_HOST_DEVICE static constexpr int compute_impl(
      const Index<IndexArgs...>& idx, const Stride<StrideArgs...>& stride,
                                    std::index_sequence<Is...>) {
    return (
        ... +
        dot_product_impl<
            typename std::tuple_element<Is, std::tuple<IndexArgs...>>::type,
            typename std::tuple_element<Is, std::tuple<StrideArgs...>>::type>::
            compute(std::get<Is>(idx.shape), std::get<Is>(stride.shape)));
  }
};

// template <typename Index, typename Stride>
// MYBLAS_HOST_DEVICE constexpr int get_linear_idx(const Index& index, const
// Stride& stride) {
//   return dot_product_impl<Index, Stride>::compute(index, stride);
// }

template <typename Stride, typename Index>
MYBLAS_HOST_DEVICE constexpr int get_linear_idx(const Index& index) {
  return dot_product_impl<Index, Stride>::compute(index, Stride{});
}

}  // namespace myblas