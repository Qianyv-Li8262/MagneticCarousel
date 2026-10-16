// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace eddy_codegen {
using Real = boost::multiprecision::cpp_dec_float_100;
// std::complex only guarantees built-in floating-point scalar types. Keep the
// few complex operations explicit so every operation is cpp_dec_float_100.
struct Complex {
    Real re, im;
    Complex(Real real=0, Real imaginary=0) : re(std::move(real)), im(std::move(imaginary)) {}
    const Real &real() const { return re; }
    const Real &imag() const { return im; }
    Complex &operator+=(const Complex &b) { re+=b.re; im+=b.im; return *this; }
};
inline Complex operator+(const Complex &a,const Complex &b) { return {a.re+b.re,a.im+b.im}; }
inline Complex operator-(const Complex &a,const Complex &b) { return {a.re-b.re,a.im-b.im}; }
inline Complex operator*(const Complex &a,const Complex &b) {
    return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};
}
inline Complex operator*(const Real &a,const Complex &b) { return {a*b.re,a*b.im}; }
inline Complex operator/(const Complex &a,const Complex &b) {
    const Real denominator=b.re*b.re+b.im*b.im;
    return {(a.re*b.re+a.im*b.im)/denominator,(a.im*b.re-a.re*b.im)/denominator};
}
using Vector = std::vector<Real>;
using Matrix = std::vector<Vector>;

inline Real norm(const Complex &z) { return sqrt(z.real()*z.real()+z.imag()*z.imag()); }

// alpha_sigma(i*omega)/V - alpha_0/V, using the e^(i*omega*t) convention.
inline Complex response(const Real &omega, const Real &mu, const Real &diffusion_time)
{
    if (omega == 0) return {0, 0};
    const Real t = sqrt(omega * diffusion_time / 2);
    const Complex x(t, -t);
    const Complex sx(sin(t)*cosh(t), -cos(t)*sinh(t));
    const Complex cx(cos(t)*cosh(t), sin(t)*sinh(t));
    const Complex j0 = sx/x;
    const Complex j1 = (sx-x*cx)/(x*x);
    const Complex alpha = Real("1.5") * (Real(2*mu+1)*j1-x*j0) / (Real(mu-1)*j1+x*j0);
    return alpha - Complex(3*(mu-1)/(mu+2), 0);
}

inline Real physical_root(int n, const Real &mu)
{
    const Real pi = boost::math::constants::pi<Real>();
    if (mu == 1) return n*pi;
    if (mu < 1) throw std::runtime_error("eddy codegen requires Mu_r >= 1");
    Real lo = n*pi, hi = (Real(n)+Real("0.5"))*pi;
    const auto f = [&](const Real &q) -> Real {
        return (q*q+mu-1)*sin(q)-(mu-1)*q*cos(q);
    };
    Real flo = f(lo);
    for (int i=0; i<320; ++i) {
        const Real mid = (lo+hi)/2;
        const Real fm = f(mid);
        if ((fm > 0) == (flo > 0)) { lo=mid; flo=fm; }
        else hi=mid;
    }
    return (lo+hi)/2;
}

// Normal equations are solved at 100 decimal digits: fast-mode columns can
// be nearly dependent over a finite frequency band. No double-precision solve.
inline Vector solve_passive(const Matrix &gram, const Vector &rhs, const std::vector<bool> &passive)
{
    std::vector<int> indices;
    for (int j=0; j<static_cast<int>(rhs.size()); ++j) if (passive[j]) indices.push_back(j);
    const int n=static_cast<int>(indices.size());
    Matrix a(n,Vector(n+1));
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) a[i][j]=gram[indices[i]][indices[j]];
        a[i][n]=rhs[indices[i]];
    }
    for (int k=0;k<n;++k) {
        int pivot=k;
        for (int i=k+1;i<n;++i) if (abs(a[i][k]) > abs(a[pivot][k])) pivot=i;
        if (abs(a[pivot][k]) < Real("1e-85")) throw std::runtime_error("singular eddy NNLS system");
        std::swap(a[k],a[pivot]);
        for (int i=k+1;i<n;++i) {
            const Real ratio=a[i][k]/a[k][k];
            for (int j=k;j<=n;++j) a[i][j]-=ratio*a[k][j];
        }
    }
    Vector z(rhs.size(),0), solution(n,0);
    for (int i=n-1;i>=0;--i) {
        Real v=a[i][n];
        for (int j=i+1;j<n;++j) v-=a[i][j]*solution[j];
        solution[i]=v/a[i][i];
        z[indices[i]]=solution[i];
    }
    return z;
}

// Lawson-Hanson active-set nonnegative least squares, using the Gram matrix.
inline Vector nnls(const Matrix &a, const Vector &b)
{
    const int k=static_cast<int>(a.front().size());
    Matrix gram(k,Vector(k,0)); Vector rhs(k,0),x(k,0);
    for (int j=0;j<k;++j) {
        for (int r=0;r<static_cast<int>(a.size());++r) rhs[j]+=a[r][j]*b[r];
        for (int l=0;l<k;++l)
            for (int r=0;r<static_cast<int>(a.size());++r) gram[j][l]+=a[r][j]*a[r][l];
    }
    std::vector<bool> passive(k,false);
    const Real eps("1e-60");
    int iterations=0;
    while (++iterations < 1000) {
        int enter=-1; Real best=eps;
        for (int j=0;j<k;++j) if (!passive[j]) {
            Real gradient=rhs[j];
            for (int l=0;l<k;++l) gradient-=gram[j][l]*x[l];
            if (gradient>best) { best=gradient; enter=j; }
        }
        if (enter<0) return x;
        passive[enter]=true;
        while (++iterations < 1000) {
            const Vector z=solve_passive(gram,rhs,passive);
            bool positive=true;
            for (int j=0;j<k;++j) if (passive[j] && z[j]<=0) positive=false;
            if (positive) { x=z; break; }
            Real step=1;
            for (int j=0;j<k;++j) if (passive[j] && z[j]<=0) {
                const Real candidate=x[j]/(x[j]-z[j]);
                if (candidate<step) step=candidate;
            }
            for (int j=0;j<k;++j) {
                x[j]+=step*(z[j]-x[j]);
                if (passive[j] && abs(x[j])<eps) { x[j]=0; passive[j]=false; }
            }
        }
    }
    throw std::runtime_error("eddy NNLS did not converge");
}

inline Complex compressed_response(const Real &omega, const Vector &tau, const Vector &a)
{
    Complex result(0,0);
    for (std::size_t j=0;j<a.size();++j) {
        const Real r=omega*tau[j];
        result+=a[j]*Complex(-r*r/(1+r*r),-r/(1+r*r));
    }
    return result;
}
} // namespace eddy_codegen
