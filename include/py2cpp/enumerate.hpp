#pragma once

#include <cstddef>   // import size_t
#include <iterator>  // import std::begin() std::end()
#include <type_traits>
#include <utility>

namespace py {

    /**
     * @brief
     *
     * @tparam T
     * @tparam decltype(std::begin(std::declval<T>()))
     * @tparam decltype(std::end(std::declval<T>()))
     * @param[in] iterable
     * @return constexpr auto
     */
    template <typename T, typename TIter = decltype(std::begin(std::declval<T>())),
              typename = decltype(std::end(std::declval<T>()))>
    constexpr auto enumerate(T&& iterable) {
        struct iterator {
            size_t i;
            TIter iter;
            auto operator!=(const iterator& other) const -> bool { return iter != other.iter; }
            void operator++() {
                ++i;
                ++iter;
            }
            auto operator*() const { return std::make_pair(i, *iter); }
            auto operator*() { return std::make_pair(i, *iter); }
        };

        struct iterable_wrapper {
            T iterable;
            auto begin() { return iterator{0, std::begin(iterable)}; }
            auto end() { return iterator{0, std::end(iterable)}; }
        };

        return iterable_wrapper{std::forward<T>(iterable)};
    }

}  // namespace py
