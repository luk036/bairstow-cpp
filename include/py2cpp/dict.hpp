#pragma once

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <unordered_map>
#include <utility>

// template <typename T> using Value_type = typename T::value_type;

namespace py {

    template <typename Iter> struct key_iterator : Iter {
        explicit key_iterator(Iter it) : Iter(it) {}
        auto operator*() const -> const auto& { return Iter::operator*().first; }
        auto operator*() -> auto& { return Iter::operator*().first; }
        auto operator++() -> key_iterator& {
            Iter::operator++();
            return *this;
        }
    };

    /**
     * @brief
     *
     * @tparam Key
     * @tparam T
     */
    template <typename Key, typename T> class dict : public std::unordered_map<Key, T> {
        using Self = dict<Key, T>;
        using Base = std::unordered_map<Key, T>;

      public:
        using value_type = std::pair<const Key, T>;

        /**
         * @brief Construct a new dict object
         *
         */
        dict() : Base{} {}

        /**
         * @brief Construct a new dict object
         *
         * @param[in] init
         */
        dict(std::initializer_list<value_type> init) : Base{init} {}

        /**
         * @brief
         *
         * @param[in] key
         * @return true
         * @return false
         */
        auto contains(const Key& key) const -> bool { return this->find(key) != this->end(); }

        /**
         * @brief
         *
         * @param[in] key
         * @param[in] default_value
         * @return T
         */
        auto get(const Key& key, const T& default_value) -> T {
            if (!contains(key)) {
                return default_value;
            }
            return (*this)[key];
        }

        /**
         * @brief
         *
         * @return auto
         */
        auto begin() const {
            using Iter = decltype(Base::begin());
            return key_iterator<Iter>{Base::begin()};
        }

        /**
         * @brief
         *
         * @return auto
         */
        auto end() const {
            using Iter = decltype(Base::end());
            return key_iterator<Iter>{Base::end()};
        }

        /**
         * @brief
         *
         * @return std::unordered_map<Key, T>&
         */
        auto items() -> Base& { return *this; }

        /**
         * @brief
         *
         * @return const std::unordered_map<Key, T>&
         */
        auto items() const -> const Base& { return *this; }

        /**
         * @brief
         *
         * @return _Self
         */
        auto copy() const -> Self { return *this; }

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator[](const Key& k) const -> const T& {
            return this->at(k);  // luk: a bug in std::unordered_map?
        }

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator[](const Key& k) -> T& { return Base::operator[](k); }

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator=(const Self&) -> Self& = delete;

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator=(Self&&) noexcept -> dict& = default;

        /**
         * @brief Move Constructor (default)
         *
         */
        dict(dict<Key, T>&&) noexcept = default;

        ~dict() = default;

        // private:
        /**
         * @brief Construct a new dict object
         *
         * Copy through explicitly the public copy() function!!!
         */
        dict(const dict<Key, T>&) = default;
    };

    /**
     * @brief
     *
     * @tparam Key
     * @tparam T
     * @param[in] key
     * @param[in] m
     * @return true
     * @return false
     */
    template <typename Key, typename T> inline auto operator<(const Key& key, const dict<Key, T>& m)
        -> bool {
        return m.contains(key);
    }

    /**
     * @brief
     *
     * @tparam Key
     * @tparam T
     * @param[in] m
     * @return size_t
     */
    template <typename Key, typename T> inline auto len(const dict<Key, T>& m) -> size_t {
        return m.size();
    }

    /**
     * @brief Template Deduction Guide
     *
     * @tparam Key
     * @tparam T
     */
    // template <typename Key, typename T>
    // dict(std::initializer_list<std::pair<const Key, T>>) -> dict<Key, T>;

    // template <class Sequence>
    // dict(const Sequence& S)
    //     -> dict<std::remove_cv_t<decltype(*std::begin(S))>, size_t>;

}  // namespace py
