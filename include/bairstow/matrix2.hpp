#pragma once

#include <utility> // import std::move

#include "vector2.hpp"

#if __cpp_constexpr >= 201304
#define CONSTEXPR14 constexpr
#else
#define CONSTEXPR14 inline
#endif

namespace numeric {
/**
 * @brief Matrix2
 *
 * @tparam T1
 * @tparam T2
 */
template <typename T1, typename T2 = T1> class Matrix2 {
  private:
    T1 _x;
    T2 _y;

  public:
    /**
     * @brief Construct a new Matrix2 object
     *
     * @param x
     * @param y
     */
    constexpr Matrix2(T1 &&x, T2 &&y) noexcept
        : _x{std::move(x)}, _y{std::move(y)} {}

    /**
     * @brief
     *
     * @return constexpr const T1&
     */
    constexpr auto x() const -> const T1 & { return this->_x; }

    /**
     * @brief
     *
     * @return constexpr const T2&
     */
    constexpr auto y() const -> const T2 & { return this->_y; }

    // /**
    //  * @brief Construct a new Matrix2 object
    //  *
    //  * @param x
    //  * @param y
    //  */
    // constexpr Matrix2(const T1& x, const T2& y) : Vector2<T1, T2>{x, y} {}

    /** @name Arithmetic operators
     *  definie +, -, *, /, +=, -=, *=, /=, etc.
     */
    ///@{

    /**
     * @brief Negate
     *
     * @return Matrix2<T1, T2>
     */
    constexpr auto operator-() const -> Matrix2<T1, T2> {
        return {-this->x(), -this->y()};
    }

    /**
     * @brief Add
     *
     * @tparam U1
     * @tparam U2
     * @param other
     * @return Matrix2<T1, T2>&
     */
    template <typename U1, typename U2>
    CONSTEXPR14 auto operator+=(const Matrix2<U1, U2> &other)
        -> Matrix2<T1, T2> & {
        this->_x += other.x();
        this->_y += other.y();
        return *this;
    }

    /**
     * @brief Substract
     *
     * @tparam U1
     * @tparam U2
     * @param other
     * @return Matrix2<T1, T2>&
     */
    template <typename U1, typename U2> //
    CONSTEXPR14 auto operator-=(const Matrix2<U1, U2> &other)
        -> Matrix2<T1, T2> & {
        this->_x -= other.x();
        this->_y -= other.y();
        return *this;
    }

    /**
     * @brief Multiply
     *
     * @tparam R
     * @param alpha
     * @return Matrix2<T1, T2>&
     */
    template <typename R>
    CONSTEXPR14 auto operator*=(const R &alpha) -> Matrix2<T1, T2> & {
        this->_x *= alpha;
        this->_y *= alpha;
        return *this;
    }

    /**
     * @brief Divide
     *
     * @tparam R
     * @param alpha
     * @return Matrix2<T1, T2>&
     */
    template <typename R>
    CONSTEXPR14 auto operator/=(const R &alpha) -> Matrix2<T1, T2> & {
        this->_x /= alpha;
        this->_y /= alpha;
        return *this;
    }

    /**
     * @brief Add
     *
     * @tparam U1
     * @tparam U2
     * @param x
     * @param y
     * @return Matrix2<T1, T2>
     */
    template <typename U1, typename U2> //
    friend constexpr auto operator+(Matrix2<T1, T2> x, const Matrix2<U1, U2> &y)
        -> Matrix2<T1, T2> {
        return std::move(x) += y;
    }

    /**
     * @brief Substract
     *
     * @tparam U1
     * @tparam U2
     * @param[in] x
     * @param[in] y
     * @return Matrix2<T1, T2>
     */
    template <typename U1, typename U2> //
    friend constexpr auto operator-(Matrix2<T1, T2> x, const Matrix2<U1, U2> &y)
        -> Matrix2<T1, T2> {
        return std::move(x) -= y;
    }

    /**
     * @brief Multiply by a scalar
     *
     * @tparam R
     * @param[in] x
     * @param[in] alpha
     * @return Matrix2<T1, T2>
     */
    template <typename R>
    friend constexpr auto operator*(Matrix2<T1, T2> x, const R &alpha)
        -> Matrix2<T1, T2> {
        return x *= alpha;
    }

    /**
     * @brief Multiply (by a scalar)
     *
     * @tparam R
     * @param[in] alpha
     * @param[in] x
     * @return Matrix2<T1, T2>
     */
    template <typename R>
    friend constexpr auto operator*(const R &alpha, Matrix2<T1, T2> x)
        -> Matrix2<T1, T2> {
        return x *= alpha;
    }

    /**
     * @brief Divide (by a scalar)
     *
     * @tparam R
     * @param[in] x
     * @param[in] alpha
     * @return Matrix2<T1, T2>
     */
    template <typename R>
    friend constexpr auto operator/(Matrix2<T1, T2> x, const R &alpha)
        -> Matrix2<T1, T2> {
        return x /= alpha;
    }

    /**
     * @brief Multiply with Vector2
     *
     * @tparam U1
     * @tparam U2
     * @param[in] other
     * @return Vector2<U1, U2>
     */
    template <typename U1, typename U2> //
    constexpr auto mdot(const Vector2<U1, U2> &other) const -> Vector2<U1, U2> {
        return {this->_x.dot(other), this->_y.dot(other)};
    }

    /**
     * @brief Det
     *
     * @tparam U1
     * @tparam U2
     * @param[in] other
     * @return double
     */
    constexpr auto det() const -> double {
        return this->x().x() * this->y().y() - this->x().y() * this->y().x();
    }

    ///@}
};
} // namespace numeric
