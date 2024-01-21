#pragma once

#include <cmath>
#include <utility>  // import std::move

#if __cpp_constexpr >= 201304
#    define CONSTEXPR14 constexpr
#else
#    define CONSTEXPR14 inline
#endif

namespace numeric {

    /**
     * @brief Vector2
     *
     */
    template <typename T1, typename T2 = T1> class Vector2 {
      public:
        T1 _x;
        T2 _y;

        /**
         * @brief Construct a new Vector2 object
         *
         * The function is a constructor for a Vector2 object that initializes its x and y values to
         * 0.
         */
        constexpr Vector2() : _x{0}, _y{0} {}

        /**
         * @brief Construct a new Vector2 object
         *
         * The function constructs a new Vector2 object with given x and y values.
         *
         * @param x The parameter `x` is of type `T1&&`, which means it is a forwarding reference.
         * It can accept any type, and it is passed as an rvalue reference. This allows the
         * constructor to efficiently move or forward the value of `x` into the `_x` member
         * variable.
         * @param y The parameter "y" is the y-coordinate of the Vector2 object. It represents the
         * vertical position of the vector in a 2D coordinate system.
         */
        constexpr Vector2(T1 x, T2 y) noexcept : _x{x}, _y{y} {}

        /**
         * @brief Construct a new Vector2 object
         *
         * The function constructs a new Vector2 object with given x and y values.
         *
         * @param x The parameter `x` is of type `T1&&`, which means it is a forwarding reference.
         * It can accept any type, and it is passed as an rvalue reference. This allows the
         * constructor to efficiently move or forward the value of `x` into the `_x` member
         * variable.
         * @param y The parameter "y" is the y-coordinate of the Vector2 object. It represents the
         * vertical position of the vector in a 2D coordinate system.
         */
        // constexpr Vector2(const T1 &x, const T2 &y) : _x{x}, _y{y} {}

        /**
         * @brief Construct a new Vector2 object
         *
         * The function constructs a new Vector2 object by copying the values from another Vector2
         * object.
         *
         * @tparam U1
         * @tparam U2
         * @param other The parameter "other" is a reference to another Vector2 object.
         */
        template <typename U1, typename U2> constexpr explicit Vector2(const Vector2<U1, U2> &other)
            : _x(other.x()), _y(other.y()) {}

        /**
         * The function returns a reference to a constant value of type T1.
         *
         * @return a reference to a constant object of type T1.
         */
        constexpr auto x() const noexcept -> const T1 & { return this->_x; }

        /**
         * The function `y()` returns a reference to a constant value of type `T2`.
         *
         * @return a reference to a constant object of type T2.
         */
        constexpr auto y() const noexcept -> const T2 & { return this->_y; }

        // /**
        //  * @brief
        //  *
        //  * @return double
        //  */
        // constexpr auto norm_inf() const -> double {
        //   return std::max(std::abs(this->_x), std::abs(this->_y));
        // }

        /**
         * The dot function calculates the dot product of two 2D vectors.
         *
         * @tparam U1
         * @tparam U2
         * @param other The parameter "other" is a reference to an object of type Vector2<U1, U2>.
         *
         * @return The `dot` function is returning the dot product of two vectors, which is a scalar
         * value of type `double`.
         */
        template <typename U1, typename U2>  //
        constexpr auto dot(const Vector2<U1, U2> &other) const -> double {
            return this->_x * other._x + this->_y * other._y;
        }

        /**
         * The cross product of two 2D vectors is calculated by multiplying their x and y components
         * and subtracting the result.
         *
         * @tparam U1
         * @tparam U2
         * @param other The parameter "other" is a reference to an object of type Vector2<U1, U2>.
         *
         * @return The function `cross` returns the result of the cross product between the current
         * vector and the `other` vector. The result is of type `double`.
         */
        template <typename U1, typename U2>  //
        constexpr auto cross(const Vector2<U1, U2> &other) const -> double {
            return this->_x * other._y - other._x * this->_y;
        }

        /** @name Arithmetic operators
         *  definie +, -, *, /, +=, -=, *=, /=, etc.
         */
        ///@{

        /**
         * The above function negates a Vector2 object.
         *
         * @return The `operator-` function returns a `Vector2` object with the negated values of
         * `_x` and `_y`.
         */
        constexpr auto operator-() const -> Vector2<T1, T2> { return {-this->_x, -this->_y}; }

        /**
         * The function `operator+=` adds the components of another Vector2 object to the current
         * Vector2 object and returns a reference to the updated object.
         *
         * @tparam U1
         * @tparam U2
         * @param other The parameter "other" is a reference to an object of type Vector2<U1, U2>.
         *
         * @return a reference to a Vector2 object.
         */
        template <typename U1, typename U2>
        CONSTEXPR14 auto operator+=(const Vector2<U1, U2> &other) -> Vector2<T1, T2> & {
            this->_x += other.x();
            this->_y += other.y();
            return *this;
        }

        /**
         * The function subtracts the x and y components of another Vector2 object from the current
         * Vector2 object.
         *
         * @tparam U1
         * @tparam U2
         * @param other The parameter "other" is a reference to an object of type Vector2<U1, U2>.
         *
         * @return a reference to a Vector2 object.
         */
        template <typename U1, typename U2>  //
        CONSTEXPR14 auto operator-=(const Vector2<U1, U2> &other) -> Vector2<T1, T2> & {
            this->_x -= other.x();
            this->_y -= other.y();
            return *this;
        }

        /**
         * The function multiplies the x and y components of a Vector2 object by a given value.
         *
         * @tparam R
         * @param alpha alpha is a constant reference to a variable of type R.
         *
         * @return The `operator*=` function returns a reference to the modified `Vector2` object.
         */
        template <typename R> CONSTEXPR14 auto operator*=(const R &alpha) -> Vector2<T1, T2> & {
            this->_x *= alpha;
            this->_y *= alpha;
            return *this;
        }

        /**
         * The function divides the x and y components of a Vector2 object by a given value.
         *
         * @tparam R
         * @param alpha The parameter "alpha" is of type R, which is a template parameter. It
         * represents the value by which the x and y components of the Vector2 object are divided.
         *
         * @return a reference to the current instance of the Vector2 class.
         */
        template <typename R> CONSTEXPR14 auto operator/=(const R &alpha) -> Vector2<T1, T2> & {
            this->_x /= alpha;
            this->_y /= alpha;
            return *this;
        }

        /**
         * The above function is a friend function that adds two Vector2 objects and returns the
         * result.
         *
         * @tparam U1
         * @tparam U2
         * @param x A Vector2 object with template types T1 and T2.
         * @param y The parameter `y` is a constant reference to a `Vector2` object with template
         * parameters `U1` and `U2`.
         *
         * @return a Vector2 object.
         */
        template <typename U1, typename U2>  //
        friend constexpr auto operator+(Vector2<T1, T2> x, const Vector2<U1, U2> &y)
            -> Vector2<T1, T2> {
            return x += y;
        }

        /**
         * The function is a friend function that subtracts two Vector2 objects and returns the
         * result.
         *
         * @tparam U1
         * @tparam U2
         * @param x A Vector2 object with template types T1 and T2.
         * @param y The parameter `y` is a constant reference to a `Vector2` object with template
         * parameters `U1` and `U2`.
         *
         * @return a Vector2 object.
         */
        template <typename U1, typename U2>  //
        friend constexpr auto operator-(Vector2<T1, T2> x, const Vector2<U1, U2> &y)
            -> Vector2<T1, T2> {
            return x -= y;
        }

        /**
         * The function multiplies a Vector2 object by a scalar value.
         *
         * @tparam R
         * @param x A Vector2 object of type T1 and T2.
         * @param alpha The parameter `alpha` is a scalar value that will be used to multiply each
         * component of the `Vector2` object `x`.
         *
         * @return a Vector2 object.
         */
        template <typename R> friend constexpr auto operator*(Vector2<T1, T2> x, const R &alpha)
            -> Vector2<T1, T2> {
            return x *= alpha;
        }

        /**
         * The above function multiplies a Vector2 object by a scalar value.
         *
         * @tparam R
         * @param alpha The parameter `alpha` is a scalar value that will be multiplied with the
         * vector `x`.
         * @param x A Vector2 object of type T1 and T2.
         *
         * @return a Vector2 object.
         */
        template <typename R> friend constexpr auto operator*(const R &alpha, Vector2<T1, T2> x)
            -> Vector2<T1, T2> {
            return x *= alpha;
        }

        /**
         * The above function divides a Vector2 object by a scalar value.
         *
         * @tparam R
         * @param x A Vector2 object of type T1 and T2.
         * @param alpha The parameter `alpha` is a scalar value that is used to divide each
         * component of the `Vector2` object `x`.
         *
         * @return a Vector2 object.
         */
        template <typename R> friend constexpr auto operator/(Vector2<T1, T2> x, const R &alpha)
            -> Vector2<T1, T2> {
            return x /= alpha;
        }

        ///@}

        /**
         * The above function overloads the << operator to output a Vector2 object in the format
         * "{x, y}".
         *
         * @tparam Stream
         * @param[out] out The parameter "out" is a reference to a Stream object. It is used to
         * output the contents of the Vector2 object to the stream.
         * @param[in] vec The parameter `vec` is a constant reference to an object of type
         * `Vector2<T1, T2>`.
         *
         * @return The return type of the `operator<<` function is `Stream&`.
         */
        template <class Stream> friend auto operator<<(Stream &out, const Vector2<T1, T2> &vec)
            -> Stream & {
            out << "{" << vec.x() << ", " << vec.y() << "}";
            return out;
        }
    };
}  // namespace numeric
