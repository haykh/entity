#if defined(OUTPUT_ENABLED)
  #include "utils/param_container.h"

  #include "enums.h"
  #include "global.h"

  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>

  #include <any>
  #include <functional>
  #include <map>
  #include <string>
  #include <type_traits>
  #include <typeindex>
  #include <typeinfo>
  #include <utility>
  #include <vector>

namespace prm {
  template <typename T, typename = void>
  struct has_to_string : std::false_type {};

  template <typename T>
  struct has_to_string<T, std::void_t<decltype(std::declval<T>().to_string())>>
    : std::true_type {};

  template <typename T>
  auto write(adios2::IO& io, const std::string& name, T var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    io.DefineAttribute(name, std::string(var.to_string()));
  }

  template <typename T>
  auto write(adios2::IO& io, const std::string& name, T var)
    -> decltype(void(T()), void()) {
    io.DefineAttribute(name, var);
  }

  template <>
  void write(adios2::IO& io, const std::string& name, bool var) {
    io.DefineAttribute(name, var ? 1 : 0);
  }

  template <>
  void write(adios2::IO& io, const std::string& name, Dimension var) {
    io.DefineAttribute(name, (unsigned short)var);
  }

  template <typename T>
  auto write_pair(adios2::IO& io, const std::string& name, std::pair<T, T> var)
    -> typename std::enable_if<has_to_string<T>::value, void>::type {
    std::vector<std::string> var_str;
    var_str.push_back(var.first.to_string());
    var_str.push_back(var.second.to_string());
    io.DefineAttribute(name, var_str.data(), var_str.size());
  }

  template <typename T>
  auto write_pair(adios2::IO& io, const std::string& name, std::pair<T, T> var)
    -> decltype(void(T()), void()) {
    std::vector<T> var_vec;
    var_vec.push_back(var.first);
    var_vec.push_back(var.second);
    io.DefineAttribute(name, var_vec.data(), var_vec.size());
  }

  template <typename T>
  auto write_vec(adios2::IO& io, const std::string& name, std::vector<T> var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    std::vector<std::string> var_str;
    for (const auto& v : var) {
      var_str.push_back(v.to_string());
    }
    io.DefineAttribute(name, var_str.data(), var_str.size());
  }

  template <typename T>
  auto write_vec(adios2::IO& io, const std::string& name, std::vector<T> var)
    -> decltype(void(T()), void()) {
    io.DefineAttribute(name, var.data(), var.size());
  }

  template <typename T>
  auto write_vec_pair(adios2::IO&                  io,
                      const std::string&           name,
                      std::vector<std::pair<T, T>> var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    std::vector<std::string> var_str;
    for (const auto& v : var) {
      var_str.push_back(v.first.to_string());
      var_str.push_back(v.second.to_string());
    }
    io.DefineAttribute(name, var_str.data(), var_str.size());
  }

  template <typename T>
  auto write_vec_pair(adios2::IO&                  io,
                      const std::string&           name,
                      std::vector<std::pair<T, T>> var)
    -> decltype(void(T()), void()) {
    std::vector<T> var_vec;
    for (const auto& v : var) {
      var_vec.push_back(v.first);
      var_vec.push_back(v.second);
    }
    io.DefineAttribute(name, var_vec.data(), var_vec.size());
  }

  template <typename T>
  auto write_vec_vec(adios2::IO&                 io,
                     const std::string&          name,
                     std::vector<std::vector<T>> var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    std::vector<std::string> var_str;
    for (const auto& vec : var) {
      for (const auto& v : vec) {
        var_str.push_back(v.to_string());
      }
    }
    io.DefineAttribute(name, var_str.data(), var_str.size());
  }

  template <typename T>
  auto write_vec_vec(adios2::IO&                 io,
                     const std::string&          name,
                     std::vector<std::vector<T>> var)
    -> decltype(void(T()), void()) {
    std::vector<T> var_vec;
    for (const auto& vec : var) {
      for (const auto& v : vec) {
        var_vec.push_back(v);
      }
    }
    io.DefineAttribute(name, var_vec.data(), var_vec.size());
  }

  template <typename T>
  auto write_dict(adios2::IO&              io,
                  const std::string&       name,
                  std::map<std::string, T> var) ->
    typename std::enable_if<has_to_string<T>::value, void>::type {
    for (const auto& [key, v] : var) {
      io.DefineAttribute(name + "_" + key, v.to_string());
    }
  }

  template <typename T>
  auto write_dict(adios2::IO&              io,
                  const std::string&       name,
                  std::map<std::string, T> var) -> decltype(void(T()), void()) {
    for (const auto& [key, v] : var) {
      io.DefineAttribute(name + "_" + key, v);
    }
  }

  std::map<std::type_index, std::function<void(adios2::IO&, const std::string&, std::any)>>
    write_functions;

  template <typename T>
  void register_write_function() {
    write_functions[std::type_index(typeid(T))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write(io, name, std::any_cast<T>(a));
      };
  }

  template <typename T>
  void register_write_function_for_pair() {
    write_functions[std::type_index(typeid(std::pair<T, T>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_pair(io, name, std::any_cast<std::pair<T, T>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector() {
    write_functions[std::type_index(typeid(std::vector<T>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_vec(io, name, std::any_cast<std::vector<T>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector_of_pair() {
    write_functions[std::type_index(typeid(std::vector<std::pair<T, T>>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_vec_pair(io, name, std::any_cast<std::vector<std::pair<T, T>>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector_of_vector() {
    write_functions[std::type_index(typeid(std::vector<std::vector<T>>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_vec_vec(io, name, std::any_cast<std::vector<std::vector<T>>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_dict() {
    write_functions[std::type_index(typeid(std::map<std::string, T>))] =
      [](adios2::IO& io, const std::string& name, std::any a) {
        write_dict(io, name, std::any_cast<std::map<std::string, T>>(a));
      };
  }

  void write_any(adios2::IO& io, const std::string& name, std::any a) {
    auto it = write_functions.find(a.type());
    if (it != write_functions.end()) {
      it->second(io, name, a);
    } else {
      throw std::runtime_error("No write function registered for this type");
    }
  }

  void Parameters::write(adios2::IO& io) const {
    register_write_function<double>();
    register_write_function<float>();
    register_write_function<int>();
    register_write_function<std::size_t>();
    register_write_function<unsigned int>();
    register_write_function<long int>();
    register_write_function<long double>();
    register_write_function<unsigned long int>();
    register_write_function<short>();
    register_write_function<bool>();
    register_write_function<unsigned short>();
    register_write_function<ntt::FldsBC>();
    register_write_function<ntt::PrtlBC>();
    register_write_function<ntt::Coord>();
    register_write_function<ntt::Metric>();
    register_write_function<ntt::SimEngine>();
    register_write_function<ntt::PrtlPusher>();
    register_write_function<Dimension>();
    register_write_function<std::string>();

    register_write_function_for_pair<double>();
    register_write_function_for_pair<float>();
    register_write_function_for_pair<int>();
    register_write_function_for_pair<std::size_t>();
    register_write_function_for_pair<unsigned int>();
    register_write_function_for_pair<long int>();
    register_write_function_for_pair<long double>();
    register_write_function_for_pair<unsigned long int>();
    register_write_function_for_pair<short>();
    register_write_function_for_pair<unsigned short>();
    register_write_function_for_pair<std::string>();
    register_write_function_for_pair<ntt::FldsBC>();
    register_write_function_for_pair<ntt::PrtlBC>();
    register_write_function_for_pair<ntt::Coord>();
    register_write_function_for_pair<ntt::Metric>();
    register_write_function_for_pair<ntt::SimEngine>();
    register_write_function_for_pair<ntt::PrtlPusher>();

    register_write_function_for_vector<double>();
    register_write_function_for_vector<float>();
    register_write_function_for_vector<int>();
    register_write_function_for_vector<std::size_t>();
    register_write_function_for_vector<unsigned int>();
    register_write_function_for_vector<long int>();
    register_write_function_for_vector<long double>();
    register_write_function_for_vector<unsigned long int>();
    register_write_function_for_vector<short>();
    register_write_function_for_vector<unsigned short>();
    register_write_function_for_vector<std::string>();
    register_write_function_for_vector<ntt::FldsBC>();
    register_write_function_for_vector<ntt::PrtlBC>();
    register_write_function_for_vector<ntt::Coord>();
    register_write_function_for_vector<ntt::Metric>();
    register_write_function_for_vector<ntt::SimEngine>();
    register_write_function_for_vector<ntt::PrtlPusher>();

    register_write_function_for_vector_of_pair<double>();
    register_write_function_for_vector_of_pair<float>();
    register_write_function_for_vector_of_pair<int>();
    register_write_function_for_vector_of_pair<std::size_t>();
    register_write_function_for_vector_of_pair<unsigned int>();
    register_write_function_for_vector_of_pair<long int>();
    register_write_function_for_vector_of_pair<long double>();
    register_write_function_for_vector_of_pair<unsigned long int>();
    register_write_function_for_vector_of_pair<short>();
    register_write_function_for_vector_of_pair<unsigned short>();
    register_write_function_for_vector_of_pair<std::string>();
    register_write_function_for_vector_of_pair<ntt::FldsBC>();
    register_write_function_for_vector_of_pair<ntt::PrtlBC>();
    register_write_function_for_vector_of_pair<ntt::Coord>();
    register_write_function_for_vector_of_pair<ntt::Metric>();
    register_write_function_for_vector_of_pair<ntt::SimEngine>();
    register_write_function_for_vector_of_pair<ntt::PrtlPusher>();

    register_write_function_for_vector_of_vector<double>();
    register_write_function_for_vector_of_vector<float>();
    register_write_function_for_vector_of_vector<int>();
    register_write_function_for_vector_of_vector<std::size_t>();
    register_write_function_for_vector_of_vector<unsigned int>();
    register_write_function_for_vector_of_vector<long int>();
    register_write_function_for_vector_of_vector<long double>();
    register_write_function_for_vector_of_vector<unsigned long int>();
    register_write_function_for_vector_of_vector<short>();
    register_write_function_for_vector_of_vector<unsigned short>();
    register_write_function_for_vector_of_vector<std::string>();
    register_write_function_for_vector_of_vector<ntt::FldsBC>();
    register_write_function_for_vector_of_vector<ntt::PrtlBC>();
    register_write_function_for_vector_of_vector<ntt::Coord>();
    register_write_function_for_vector_of_vector<ntt::Metric>();
    register_write_function_for_vector_of_vector<ntt::SimEngine>();
    register_write_function_for_vector_of_vector<ntt::PrtlPusher>();

    register_write_function_for_dict<double>();
    register_write_function_for_dict<float>();
    register_write_function_for_dict<int>();
    register_write_function_for_dict<std::size_t>();
    register_write_function_for_dict<unsigned int>();
    register_write_function_for_dict<long int>();
    register_write_function_for_dict<long double>();
    register_write_function_for_dict<unsigned long int>();
    register_write_function_for_dict<short>();
    register_write_function_for_dict<unsigned short>();
    register_write_function_for_dict<std::string>();

    for (auto& [key, value] : allVars()) {
      try {
        write_any(io, key, value);
      } catch (const std::exception& e) {
        continue;
      }
    }
  }

} // namespace prm

#endif
