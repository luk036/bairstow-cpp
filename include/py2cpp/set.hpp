#pragma once

#include <cstddef>  // import size_t
#include <initializer_list>
#include <type_traits>
#include <unordered_set>
#include <utility>

// template <typename T> using Value_type = typename T::value_type;

namespace py {

    /**
     * @brief
     *
     * @tparam Key
     */
    template <typename Key> class set : public std::unordered_set<Key> {
        using Self = set<Key>;
        using Base = std::unordered_set<Key>;

      public:
        /**
         * @brief Construct a new set object
         *
         */
        set() : Base{} {}

        /**
         * @brief Construct a new set object
         *
         */
        template <typename FwdIter> set(const FwdIter& start, const FwdIter& stop)
            : Base(start, stop) {}

        /**
         * @brief Construct a new set object
         *
         * @param[in] init
         */
        set(std::initializer_list<Key> init) : Base{init} {}

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
         * @return _Self
         */
        auto copy() const -> set { return *this; }

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator=(const set&) -> set& = delete;

        /**
         * @brief
         *
         * @return _Self&
         */
        auto operator=(set&&) noexcept -> set& = default;

        /**
         * @brief Move Constructor (default)
         *
         */
        set(set<Key>&&) noexcept = default;

        // private:
        /**
         * @brief Copy Constructor (deleted)
         *
         * Copy through explicitly the public copy() function!!!
         */
        set(const set<Key>&) = default;
    };

    /**
     * @brief
     *
     * @tparam Key
     * @param[in] key
     * @param[in] m
     * @return true
     * @return false
     */
    template <typename Key> inline auto operator<(const Key& key, const set<Key>& m) -> bool {
        return m.contains(key);
    }

    /**
     * @brief
     *
     * @tparam Key
     * @param[in] m
     * @return size_t
     */
    template <typename Key> inline auto len(const set<Key>& m) -> size_t { return m.size(); }

    /**
     * @brief Template Deduction Guide
     *
     * @tparam Key
     */
    // template <typename Key>
    // set(std::initializer_list<Key>) -> set<Key>;

    // template <typename Key>
    // set(std::initializer_list<const char*> ) -> set<std::string>;
}  // namespace py
