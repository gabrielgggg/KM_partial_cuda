// shared.h -- per-node kernels used by both back ends (residuals, Thomas factors, simulation, impulse responses).
#pragma once
#include "model.h"

// residuals of the HJB and pricing equations at node k (difference form of the upwind generator)
template <typename T>
HD void resid_node(int i, int j, int nB, int nz, const T* W, const T* q, const T* S, const T* u, const T* d,
                   const T* hf, const T* hb, const T* zup, const T* zdn, const Par<T>& p, T Vref, T& rV, T& rq) {
  const int k = i * nz + j;
  const T s = S[k], sp = s > T(0) ? s : T(0), sm = s < T(0) ? s : T(0);
  T AW = T(0), Aq = T(0);
  if (i < nB - 1) { AW += sp / hf[i] * (W[k + nz] - W[k]); Aq += sp / hf[i] * (q[k + nz] - q[k]); }
  if (i > 0) { AW += -sm / hb[i] * (W[k - nz] - W[k]); Aq += -sm / hb[i] * (q[k - nz] - q[k]); }
  if (j < nz - 1) { AW += zup[j] * (W[k + 1] - W[k]); Aq += zup[j] * (q[k + 1] - q[k]); }
  if (j > 0) { AW += zdn[j] * (W[k - 1] - W[k]); Aq += zdn[j] * (q[k - 1] - q[k]); }
  const T coupon = p.delta + p.lam;
  const T xi = p.r + p.delta - p.kappa * coupon * d[k];
  rV = fabs(p.rho * W[k] + p.rho * Vref - u[k] - AW);
  rq = fabs(xi * q[k] - (T(1) - d[k]) * coupon - Aq);
}

// Thomas factors of the endowment-direction tridiagonal system of debt row i:
//   -lo[j] x[j-1] + D[j] x[j] - up[j] x[j+1] = r[j];   y_j = inv_j r_j + m_j y_{j-1};  x_j = y_j - cp_j x_{j+1}
template <typename T>
HD void thomas_factor(int nz, const T* D, const T* lo, const T* up, T* m, T* inv, T* cp) {
  T den = D[0];
  inv[0] = T(1) / den; cp[0] = -up[0] * inv[0]; m[0] = T(0);
  for (int j = 1; j < nz; ++j) {
    den = D[j] + lo[j] * cp[j - 1];
    inv[j] = T(1) / den; cp[j] = -up[j] * inv[j]; m[j] = lo[j] * inv[j];
  }
}

// ---------------------------------------------------------------- simulation (annual accounting, pdmoments.simulate)
template <typename T> struct SimArgs {
  const T *S, *d, *c, *q, *Bg, *zg, *cdf;
  int nB, nz, N;
  T mu, sigma, delta_lam, r, delta, g0, g1, g2, zt, Fmin;
  int P, nseg, L, burn, spy, Y;      // paths, segments per path, years per segment, burn-in years, steps per year, years per path
  int paper_start; T B0, z0;          // paper layout: fixed start (B0, z0); otherwise a draw from the ergodic distribution
  unsigned seed;
  T* out;                             // [8][P][Y]: y, cons, debt_due, debt_service, def_coupons, pd, B, spread
};

template <typename T> HD void interp_w(const SimArgs<T>& a, T B, T z, int& i, int& j, T& wB, T& wz) {
  const T h = a.Bg[1] - a.Bg[0];
  i = (int)floor((B - a.Bg[0]) / h);
  if (i < 0) i = 0; if (i > a.nB - 2) i = a.nB - 2;
  wB = (B - a.Bg[i]) / (a.Bg[i + 1] - a.Bg[i]);
  if (wB < T(0)) wB = T(0); if (wB > T(1)) wB = T(1);
  const T dz = a.zg[1] - a.zg[0];
  j = (int)((z - a.zg[0]) / dz);
  if (j < 0) j = 0; if (j > a.nz - 2) j = a.nz - 2;
  wz = (z - a.zg[j]) / dz;
  if (wz < T(0)) wz = T(0); if (wz > T(1)) wz = T(1);
}
template <typename T> HD T interp_v(const T* F, int nz, int i, int j, T wB, T wz) {
  const int k = i * nz + j;
  return (T(1) - wB) * (T(1) - wz) * F[k] + wB * (T(1) - wz) * F[k + nz] + (T(1) - wB) * wz * F[k + 1] + wB * wz * F[k + nz + 1];
}

template <typename T> HD void sim_segment(int tid, const SimArgs<T>& a) {
  const int p = tid / a.nseg, sg = tid % a.nseg;
  uint32_t ctr = 0;
  const uint32_t key0 = a.seed, key1 = 0x5EEDu;
  T B, z;
  if (a.paper_start) { B = a.B0; z = a.z0; }
  else {
    U4 r = philox(U4{0xFFFFFFFFu, (uint32_t)tid, 0u, 0u}, key0, key1);
    T uu = u01<T>(r.x);
    int lo = 0, hi = a.N - 1;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a.cdf[mid] < uu) lo = mid + 1; else hi = mid; }
    B = a.Bg[lo / a.nz]; z = a.zg[lo % a.nz];
  }
  const T dt = T(1) / T(a.spy), sdt = sqrt(dt);
  const T zmin = a.zg[0], zmax = a.zg[a.nz - 1], Bmin = a.Bg[0], Bmax = a.Bg[a.nB - 1];
  const T ezt = exp(a.zt);
  T nrm[4]; int nn = 4;
  const long PY = (long)a.P * a.Y;
  for (int t = -a.burn; t < a.L; ++t) {
    T y_acc = 0, c_acc = 0, dd_acc = 0, ds_acc = 0, dc_acc = 0;
    for (int s = 0; s < a.spy; ++s) {
      int i, j; T wB, wz;
      interp_w(a, B, z, i, j, wB, wz);
      const T Sv = interp_v(a.S, a.nz, i, j, wB, wz);
      const T dv = interp_v(a.d, a.nz, i, j, wB, wz);
      const T cv = interp_v(a.c, a.nz, i, j, wB, wz);
      T ez = exp(z), y = ez;
      if (dv > T(0)) {
        T F = T(1);
        if (z >= a.zt) { T v = T(1) - (ez - ezt) * a.g2; F = v > a.Fmin ? v : a.Fmin; }
        y = (T(1) - a.g0 * pow(dv, a.g1)) * F * ez;
      }
      const T due = a.delta_lam * B;
      y_acc += y * dt; c_acc += cv * dt; dd_acc += due * dt;
      ds_acc += (T(1) - dv) * due * dt; dc_acc += dv * due * dt;
      B = B + Sv * dt;
      if (B < Bmin) B = Bmin;
      if (B > Bmax) B = Bmax;
      if (nn == 4) {
        U4 r = philox(U4{ctr++, (uint32_t)tid, 1u, 0u}, key0, key1);
        T u1 = u01<T>(r.x), u2 = u01<T>(r.y), u3 = u01<T>(r.z), u4 = u01<T>(r.w);
        T r1 = sqrt(-T(2) * log(u1)), r2 = sqrt(-T(2) * log(u3));
        const T TWO_PI = T(6.283185307179586);
        nrm[0] = r1 * cos(TWO_PI * u2); nrm[1] = r1 * sin(TWO_PI * u2);
        nrm[2] = r2 * cos(TWO_PI * u4); nrm[3] = r2 * sin(TWO_PI * u4);
        nn = 0;
      }
      z = z - a.mu * z * dt + a.sigma * sdt * nrm[nn++];
      if (z < zmin) z = T(2) * zmin - z;
      if (z > zmax) z = T(2) * zmax - z;
    }
    if (t >= 0) {
      int i, j; T wB, wz;
      interp_w(a, B, z, i, j, wB, wz);
      const T qv = interp_v(a.q, a.nz, i, j, wB, wz);
      const long o = (long)p * a.Y + (long)sg * a.L + t;
      a.out[0 * PY + o] = y_acc; a.out[1 * PY + o] = c_acc; a.out[2 * PY + o] = dd_acc; a.out[3 * PY + o] = ds_acc;
      a.out[4 * PY + o] = dc_acc; a.out[5 * PY + o] = dd_acc > T(1e-12) ? dc_acc / dd_acc : T(0);
      a.out[6 * PY + o] = B; a.out[7 * PY + o] = a.delta_lam / qv - (a.delta + a.r);
    }
  }
}

// ---------------------------------------------------------------- impulse responses (Crank-Nicolson on dg/dt = A^T g)
// coefficients of row k of A^T: on g[i-1], g[i+1], and the diagonal of A
template <typename T>
HD void at_coefs(int i, int j, int nB, int nz, const T* S, const T* hf, const T* hb, const T* zup, const T* zdn,
                 T& cLo, T& cHi, T& diagA) {
  const int k = i * nz + j;
  cLo = (i > 0) ? (S[k - nz] > T(0) ? S[k - nz] : T(0)) / hf[i - 1] : T(0);
  cHi = (i < nB - 1) ? -(S[k + nz] < T(0) ? S[k + nz] : T(0)) / hb[i + 1] : T(0);
  const T s = S[k];
  diagA = -(s > T(0) ? s : T(0)) / hf[i] + (s < T(0) ? s : T(0)) / hb[i] - zup[j] - zdn[j];
}
// (A^T g)_k
template <typename T>
HD T at_apply(int i, int j, int nB, int nz, const T* S, const T* hf, const T* hb, const T* zup, const T* zdn, const T* g) {
  T cLo, cHi, dA;
  at_coefs(i, j, nB, nz, S, hf, hb, zup, zdn, cLo, cHi, dA);
  const int k = i * nz + j;
  T v = dA * g[k];
  if (i > 0) v += cLo * g[k - nz];
  if (i < nB - 1) v += cHi * g[k + nz];
  if (j > 0) v += zup[j - 1] * g[k - 1];
  if (j < nz - 1) v += zdn[j + 1] * g[k + 1];
  return v;
}
// solve debt row i of (I - h A^T) x = rhs exactly in the endowment direction, with rows i-1, i+1 taken from x
template <typename T>
HD void irf_row(int i, int nB, int nz, const T* S, const T* hf, const T* hb, const T* m, const T* inv, const T* cp,
                const T* rhs, T* x, T h) {
  const int k0 = i * nz;
  T yprev = T(0);
  // forward pass stores y in x (row i is overwritten only after its neighbours' values are read)
  T ybuf[256];
  for (int j = 0; j < nz; ++j) {
    const int k = k0 + j;
    T r = rhs[k];
    if (i > 0) { T s = S[k - nz]; if (s > T(0)) r += h * s / hf[i - 1] * x[k - nz]; }
    if (i < nB - 1) { T s = S[k + nz]; if (s < T(0)) r += -h * s / hb[i + 1] * x[k + nz]; }
    yprev = inv[k] * r + m[k] * yprev;
    ybuf[j] = yprev;
  }
  T xn = T(0);
  for (int j = nz - 1; j >= 0; --j) { xn = ybuf[j] - cp[k0 + j] * xn; x[k0 + j] = xn; }
}
