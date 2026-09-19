## lut gen
use boost::multiprecision to generate field LUT
magnetic potential is sum of all the dipoles.
Expansion formula:

$$
\boxed{
\Phi(\rho,\varphi,a)
=
\frac{2Cna}{(\rho R)^{3/2}}
\sum_{\ell=0}^{\infty}
t^{\,m_\ell+\frac32}
\frac{\left(\frac32\right)_{m_\ell}}{m_\ell!}\,
{}_2F_1\!\left(
m_\ell+\frac32,\,
\frac32;\,
m_\ell+1;\,
t^2
\right)
\cos\!\left[m_\ell(\varphi)\right]
}
$$

$$
m_\ell=\frac{(2\ell+1)n}{2},
\qquad
\chi=\frac{a^2+\rho^2+R^2}{2\rho R},
\qquad
t=\chi-\sqrt{\chi^2-1}=\frac{1}{\chi+\sqrt{\chi^2-1}}.
$$

n is the number of magnets

C is $\frac{\mu_0 m}{4 \pi}$

assume the first dipole($+z$) is on $\theta=0$