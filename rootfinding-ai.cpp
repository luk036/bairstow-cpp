#include <cmath>
#include <vector>

class Vector2 {
  public:
    double x double y;
    Vector2(double x = 0, double y = 0) : x(x), y(y) {}
    Vector2 operator+(const Vector2 &other) const { return Vector2(x + other.x, y + other.y); }
    Vector2 operator-(const Vector2 &other) const { return Vector2(x - other.x, y - other.y); }
    Vector2 operator*(double scalar) const { return Vector2(x * scalar, y * scalar); }
    Vector2 operator/(double scalar) const { return Vector2(x / scalar, y / scalar); }
    double operator*(const Vector2 &other) const { return x * other.x + y * other.y; }
    double length() const { return std::sqrt(x * x + y * y); }
    double length2() const { return x * x + y * y; }
    Vector2 normalize() const { return *this / length(); }
};

class Matrix2 {
  public:
    std::array<Vector2, 2> rows;

    Matrix2(Vector2 row1 = Vector2(), Vector2 row2 = Vector2()) {
        rows[0] = row1;
        rows[1] = row2;
    }
    Matrix2 operator*(const Matrix2 &other) const {
        Matrix2 result;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result.rows[i].x += rows[i].y * other.rows[j].x;
                result.rows[i].y += rows[i].y * other.rows[j].y;
            }
        }
        return result;
    }
    Vector2 mdot(const Vector2 &vec) const { return Vector2(rows[0] * vec, rows[1] * vec); }
    double det() const { return rows[0].x * rows[1].y - rows[0].y * rows[1].x; }
};

class VdCorput {
  public:
    int base;
    int seed;
    VdCorput(int base = 2, int seed = 0) : base(base), seed(seed) {}
    void reseed(int seed) { this->seed = seed; }
    double pop() {
        double r = 0.0;
        double base_inv = 1.0 / base;
        double f = base_inv;
        int n = seed;
        while (n > 0) {
            r += (n % base) * f;
            f *= base_inv;
            n /= base;
        }
        seed++;
        return r;
    }
};

class Robin {
  public:
    int M;
    std::vector<int> exclude_list;
    Robin(int M) : M(M) { exclude_list.reserve(M); }
    std::vector<int>::iterator begin() { return exclude_list.begin(); }
    std::vector<int>::iterator end() { return exclude_list.end(); }
    void exclude(int i) {
        exclude_list.clear();
        for (int j = 0; j < M; j++) {
            if (j != i) {
                exclude_list.push_back(j);
            }
        }
    }
};

const double PI = std::acos(-1.0);

class Options {
  public:
    int max_iters = 2000;
    double tolerance = 1e-12;
    double tol_ind = 1e-15;
};

Vector2 delta(const Vector2 &vA, const Vector2 &vr, const Vector2 &vp) {
    double r = vr.x, q = vr.y;
    double p = vp.x, s = vp.y;
    Matrix2 mp(Vector2(s, -p), Vector2(-p * q, p * r + s));
    return mp.mdot(vA) / mp.det();
}

void suppress_old(Vector2 &vA, Vector2 &vA1, const Vector2 &vri, const Vector2 &vrj) {
    double A = vA.x, B = vA.y;
    double A1 = vA1.x, B1 = vA1.y;
    Vector2 vp = vri - vrj;
    double r = vri.x, q = vri.y;
    double p = vp.x, s = vp.y;
    double f = r * p + s;
    double qp = q * p;
    double e = f * s - qp * p;
    double a = A * s - B * p;
    double b = B * f - A * qp;
    double c = A1 * e - a;
    double d = B1 * e - b - a * p;
    vA.x = a * e;
    vA.y = b * e;
    vA1.x = c * s - d * p;
    vA1.y = d * f - c * qp;
}

std::pair<Vector2, Vector2> suppress(const Vector2 &vA, const Vector2 &vA1, const Vector2 &vri,
                                     const Vector2 &vrj) {
    Vector2 vp = vri - vrj;
    double r = vri.x, q = vri.y;
    double p = vp.x, s = vp.y;
    Matrix2 m_adjoint(Vector2(s, -p), Vector2(-p * q, p * r + s));
    double e = m_adjoint.det();
    Vector2 va = m_adjoint.mdot(vA);
    Vector2 vc = vA1 * e - va;
    vc.y -= va.x * p;
    va *= e;
    Vector2 va1 = m_adjoint.mdot(vc);
    return {va, va1};
}

double horner_eval(std::vector<double> coeffs, int degree, double zval) {
    for (int i = 0; i < degree; i++) {
        coeffs[i + 1] += coeffs[i] * zval;
    }
    return coeffs[degree];
}

Vector2 horner(const std::vector<double> &coeffs, int degree, const Vector2 &vr) {
    std::vector<double> temp_coeffs(coeffs);
    for (int i = 0; i < degree - 1; i++) {
        temp_coeffs[i + 1] += temp_coeffs[i] * vr.x;
        temp_coeffs[i + 2] += temp_coeffs[i] * vr.y;
    }
    return Vector2(temp_coeffs[degree - 1], temp_coeffs[degree]);
}

std::vector<Vector2> initial_guess(const std::vector<double> &coeffs) {
    int degree = coeffs.size() - 1;
    double centroid = -coeffs[1] / (degree * coeffs[0]);

    double Pc = horner_eval(coeffs, degree, centroid);
    double reff = std::pow(std::abs(Pc), 1.0 / degree);
    double m = centroid * centroid + reff * reff;
    std::vector<Vector2> vr0s;
    degree /= 2;
    degree *= 2;

    VdCorput vgen(2);
    vgen.reseed(1);
    for (int i = 1; i < degree; i += 2) {
        double temp = reff * std::cos(PI * vgen.pop());
        double r0 = 2 * (centroid + temp);
        double t0 = m + 2 * centroid * temp;
        vr0s.emplace_back(r0, -t0);
    }
    return vr0s;
}

std::pair<std::vector<Vector2>, int, bool> pbairstow_even(const std::vector<double> &coeffs,
                                                          const std::vector<Vector2> &vrs,
                                                          const Options &options = Options()) {
    int M = vrs.size();
    int N = coeffs.size() - 1;
    std::vector<bool> converged(M, false);
    Robin robin(M);
    for (int niter = 0; niter < options.max_iters; niter++) {
        double tolerance = 0.0;

        for (int i = 0; i < M; i++) {
            if (converged[i]) {
                continue;
            }
            std::vector<double> coeffs1(coeffs);
            Vector2 vA = horner(coeffs1, N, vrs[i]);
            double tol_i = std::max(std::abs(vA.x), std::abs(vA.y));
            if (tol_i < options.tol_ind) {
                converged[i] = true;
                continue;
            }
            Vector2 vA1 = horner(coeffs1, N - 2, vrs[i]);
            tolerance = std::max(tol_i, tolerance);

            for (int j : robin.exclude_list) {
                std::tie(vA, vA1) = suppress(vA, vA1, vrs[i], vrs[j]);
            }

            vrs[i] -= delta(vA, vrs[i], vA1);
        }
        if (tolerance < options.tolerance) {
            return {vrs, niter, true};
        }
    }
    return {vrs, options.max_iters, false};
}

std::pair<double | std::complex<double>, double | std::complex<double>> find_rootq(
    const Vector2 &vr) {
    double hr = vr.x / 2;
    double d = hr * hr + vr.y;
    if (d < 0) {
        double x1 = hr + std::sqrt(-d) * 1i;
        double x2 = hr - std::sqrt(-d) * 1i;
        return {x1, x2};
    } else {
        double x1 = hr + ((hr >= 0) ? std::sqrt(d) : -std::sqrt(d));
        double x2 = -vr.y / x1;
        return {x1, x2};
    }
}

Vector2 extract_autocorr(const Vector2 &vr) {
    double r = vr.x, q = vr.y;
    double hr = r / 2.0;
    double d = hr * hr + q;
    if (d < 0.0) {
        if (q < -1.0) {
            return Vector2(-r, 1.0) / q;
        }
    } else {
        double a1 = hr + ((hr >= 0.0) ? std::sqrt(d) : -std::sqrt(d));
        double a2 = -q / a1;
        if (std::abs(a1) > 1.0) {
            if (std::abs(a2) > 1.0) {
                a2 = 1.0 / a2;
            }
            a1 = 1.0 / a1;
            return Vector2(a1 + a2, -a1 * a2);
        } else if (std::abs(a2) > 1.0) {
            a2 = 1.0 / a2;
            return Vector2(a1 + a2, -a1 * a2);
        }
    }
    return vr;
}

void test_rootfind() {
    std::vector<double> h = {5.0, 2.0, 9.0, 6.0, 2.0};
    std::vector<Vector2> vr0s = initial_guess(h);
    auto [_, niter, found] = pbairstow_even(h, vr0s);
}
