#pragma once
#include <tuple>
#include <utility>
#include <cstddef>
#include <type_traits>
#include <iostream>

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

  constexpr ShapeBase() : shape() {}

  constexpr ShapeBase(Args... args) : shape(std::make_tuple(args...)) {}
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
constexpr Shape<Args...> make_shape(Args... args) {
  return Shape<Args...>(args...);
}

template <typename... Args>
constexpr Stride<Args...> make_stride(Args... args) {
  return Stride<Args...>(args...);
}

template <typename... Args>
constexpr Index<Args...> make_index(Args... args) {
  return Index<Args...>(args...);
}

void print_index(const DynamicInt& d) { std::cout << d.value << " "; }

template <typename T>
void print_element(const T& t) {
  if constexpr (std::is_integral_v<T>) {
    std::cout << t << " ";
  } else if constexpr (std::is_same_v<T, DynamicInt>) {
    print_index(t);
  } else {
    print_index(t);
  }
}

template <typename Tuple, std::size_t... Is>
void print_tuple_impl(const Tuple& t, std::index_sequence<Is...>) {
  (print_element(std::get<Is>(t)), ...);
}

template <typename... Args>
void print_index(const std::tuple<Args...>& t) {
  print_tuple_impl(
      t, std::make_index_sequence<std::tuple_size_v<std::tuple<Args...>>>{});
  std::cout << std::endl;
}

template <typename... Args>
void print_index(const Index<Args...>& s) {
  print_index(s.shape);
}

template <typename T>
struct to_dynamic_int_impl;

template <int N>
struct to_dynamic_int_impl<Int<N>> {
  static constexpr DynamicInt convert() { return DynamicInt{N}; }
};

template <int... Ns>
struct to_dynamic_int_impl<Shape<Int<Ns>...>> {
  static constexpr auto convert() {
    return make_index(to_dynamic_int_impl<Int<Ns>>::convert()...);
  }
};

template <typename... Shapes>
struct to_dynamic_int_impl<Shape<Shapes...>> {
  static constexpr auto convert() {
    return make_index(to_dynamic_int_impl<Shapes>::convert()...);
  }
};

template <typename Shape>
constexpr auto to_dynamic_int() {
  return to_dynamic_int_impl<Shape>::convert();
}

template <typename Shape, typename Stride>
struct reshape_impl;

template <int... Ns, int... Ms>
struct reshape_impl<Shape<Int<Ns>...>, Stride<Int<Ms>...>> {
  static constexpr auto compute(const int tid) {
    return make_index(
        (Ms == 0 ? DynamicInt{0} : DynamicInt{(tid / Ms) % Ns})...);
  }
};

template <typename... ShapeTs, typename... StrideTs>
struct reshape_impl<Shape<ShapeTs...>, Stride<StrideTs...>> {
  static constexpr auto compute(const int tid) {
    return make_index(reshape_impl<ShapeTs, StrideTs>::compute(tid)...);
  }
};

template <int N, int M>
struct reshape_impl<Int<N>, Int<M>> {
  static constexpr DynamicInt compute(const int tid) {
    return M == 0 ? DynamicInt{0} : DynamicInt{(tid / M) % N};
  }
};

template <typename Shape, typename Stride>
constexpr auto reshape(const int tid) {
  return reshape_impl<Shape, Stride>::compute(tid);
}

template <typename Index, typename Stride>
struct dot_product_impl;

template <int N>
struct dot_product_impl<DynamicInt, Int<N>> {
  static constexpr int compute(const DynamicInt& idx, const Int<N>&) {
    return N * idx.value;
  }
};

template <typename... IndexArgs, typename... StrideArgs>
struct dot_product_impl<Index<IndexArgs...>, Stride<StrideArgs...>> {
  static constexpr int compute(const Index<IndexArgs...>& idx,
                               const Stride<StrideArgs...>& stride) {
    return compute_impl(idx, stride, std::index_sequence_for<IndexArgs...>{});
  }

 private:
  template <std::size_t... Is>
  static constexpr int compute_impl(const Index<IndexArgs...>& idx,
                                    const Stride<StrideArgs...>& stride,
                                    std::index_sequence<Is...>) {
    return (
        ... +
        dot_product_impl<
            typename std::tuple_element<Is, std::tuple<IndexArgs...>>::type,
            typename std::tuple_element<Is, std::tuple<StrideArgs...>>::type>::
            compute(std::get<Is>(idx.shape), std::get<Is>(stride.shape)));
  }
};

template <typename Index, typename Stride>
constexpr int get_linear_idx(const Index& index, const Stride& stride) {
  return dot_product_impl<Index, Stride>::compute(index, stride);
}

template <typename Stride, typename Index>
constexpr int get_linear_idx(const Index& index) {
  return dot_product_impl<Index, Stride>::compute(index, Stride{});
}

} // namespace myblas