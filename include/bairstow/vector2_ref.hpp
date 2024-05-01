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
     * @brief Vector2Ref
     *
     */
    class Vector2Ref {
      public:
        double &_x;
        double &_y;

        /**
         * @brief Construct a new Vector2Ref object
         *
         * The function constructs a new Vector2Ref object with given x and y values.
         *
         * @param[in] x The parameter `x` is of type `double&&`, which means it is a forwarding
         * reference. It can accept any type, and it is passed as an rvalue reference. This allows
         * the constructor to efficiently move or forward the value of `x` into the `_x` member
         * variable.
         * @param[in] y The parameter "y" is the y-coordinate of the Vector2Ref object. It
         * represents the vertical position of the vector in a 2D coordinate system.
         */
        constexpr Vector2Ref(double &x, double &y) noexcept : _x{x}, _y{y} {}

        /**
         * The function returns a reference to a constant value of type double.
         *
         * @return a reference to a constant object of type double.
         */
        constexpr auto x() const noexcept -> const double & { return this->_x; }

        /**
         * The function `y()` returns a reference to a constant value of type `double`.
         *
         * @return a reference to a constant object of type double.
         */
        constexpr auto y() const noexcept -> const double & { return this->_y; }

        /**
         * The dot function calculates the dot product of two 2D vectors.
         *
         * @tparam U1
         * @tparam U2
         * @param[in] other The parameter "other" is a reference to an object of type Vector2Ref<U1,
         * U2>.
         *
         * @return The `dot` function is returning the dot product of two vectors, which is a scalar
         * value of type `double`.
         */
        constexpr auto dot(const Vector2Ref &other) const -> double {
            return this->_x * other._x + this->_y * other._y;
        }

        /**
         * The cross product of two 2D vectors is calculated by multiplying their x and y components
         * and subtracting the result.
         *
         * @tparam U1
         * @tparam U2
         * @param[in] other The parameter "other" is a reference to an object of type Vector2Ref<U1,
         * U2>.
         *
         * @return The function `cross` returns the result of the cross product between the current
         * vector and the `other` vector. The result is of type `double`.
         */
        constexpr auto cross(const Vector2Ref &other) const -> double {
            return this->_x * other._y - other._x * this->_y;
        }

        /** @name Arithmetic operators
         *  definie +, -, *, /, +=, -=, *=, /=, etc.
         */
        ///@{

        /**
         * The function `operator+=` adds the components of another Vector2Ref object to the current
         * Vector2Ref object and returns a reference to the updated object.
         *
         * @tparam U1
         * @tparam U2
         * @param[in] other The parameter "other" is a reference to an object of type Vector2Ref<U1,
         * U2>.
         *
         * @return a reference to a Vector2Ref object.
         */
        CONSTEXPR14 auto operator+=(const Vector2Ref &other) -> Vector2Ref & {
            this->_x += other.x();
            this->_y += other.y();
            return *this;
        }

        /**
         * The function subtracts the x and y components of another Vector2Ref object from the
         * current Vector2Ref object.
         *
         * @tparam U1
         * @tparam U2
         * @param[in] other The parameter "other" is a reference to an object of type Vector2Ref<U1,
         * U2>.
         *
         * @return a reference to a Vector2Ref object.
         */
        CONSTEXPR14 auto operator-=(const Vector2Ref &other) -> Vector2Ref & {
            this->_x -= other.x();
            this->_y -= other.y();
            return *this;
        }

        /**
         * The function multiplies the x and y components of a Vector2Ref object by a given value.
         *
         * @tparam R
         * @param[in] alpha alpha is a constant reference to a variable of type R.
         *
         * @return The `operator*=` function returns a reference to the modified `Vector2Ref`
         * object.
         */
        CONSTEXPR14 auto operator*=(const double &alpha) -> Vector2Ref & {
            this->_x *= alpha;
            this->_y *= alpha;
            return *this;
        }

        /**
         * The function divides the x and y components of a Vector2Ref object by a given value.
         *
         * @tparam R
         * @param[in] alpha The parameter "alpha" is of type R, which is a template parameter. It
         * represents the value by which the x and y components of the Vector2Ref object are
         * divided.
         *
         * @return a reference to the current instance of the Vector2Ref class.
         */
        CONSTEXPR14 auto operator/=(const double &alpha) -> Vector2Ref & {
            this->_x /= alpha;
            this->_y /= alpha;
            return *this;
        }

        ///@}

        /**
         * The above function overloads the << operator to output a Vector2Ref object in the format
         * "{x, y}".
         *
         * @tparam Stream
         * @param[out] out The parameter "out" is a reference to a Stream object. It is used to
         * output the contents of the Vector2Ref object to the stream.
         * @param[in] vec The parameter `vec` is a constant reference to an object of type
         * `Vector2Ref`.
         *
         * @return The return type of the `operator<<` function is `Stream&`.
         */
        template <class Stream>
        friend auto operator<<(Stream &out, const Vector2Ref &vec) -> Stream & {
            out << "{" << vec.x() << ", " << vec.y() << "}";
            return out;
        }
    };
}  // namespace numeric
