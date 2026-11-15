// model.h -- primitives of the continuous-time partial-default model (Kim and Mihalache), shared by the CPU
// (OpenMP) and GPU (CUDA) code paths. Every function is __host__ __device__ and templated on the float type T.
// The node problem mirrors pdmodel.best_S / best_node / policies of the replication package line by line.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

template <typename T> struct Par {
  T nu, rho, delta, kappa, r, lam, g0, g1, g2, zt, cmf, Scap, Fmin, qf, qband, lband;
  int closed_d;
  int n_golden;
};

template <typename T> HD T NEGH() { return T(-1.0e30); }

template <typename T> HD T u_fn(T c, T nu) {
  if (nu == T(2)) return -T(1) / c;
  if (nu == T(1)) return log(c);
  return pow(c, T(1) - nu) / (T(1) - nu);
}

// underwriting standard: returns capped flag, writes the cap L
template <typename T> HD bool issue_cap(T q, const Par<T>& p, T& L) {
  if (q < p.qf) { L = T(0); return true; }
  if (p.qband > T(0) && p.qf > T(0) && q < p.qf + p.qband) {
    T w = (q - p.qf) / p.qband;
    L = p.lband * w / (T(1) - w);
    return true;
  }
  L = T(0);
  return false;
}

// share of the endowment kept at default intensity d; Fz = fixed-cost factor of the node's z (precomputed)
template <typename T> HD T phi_fn(T d, T Fz, const Par<T>& p) {
  if (d <= T(0)) return T(1);
  return (T(1) - p.g0 * pow(d, p.g1)) * Fz;
}

struct HSC_ {};
template <typename T> struct Cand { T H, S, c, d; };

// maximize u(c(S,d)) + S VB over admissible S for given d; mode 1: S>=0, -1: S<=0, 0: S=0, 2: free
template <typename T>
HD void best_S(T d, T VB, T q, T B, T y, T Fz, const Par<T>& p, int mode, T& H, T& S, T& c) {
  const T ph = phi_fn(d, Fz, p);
  const T caut = ph * y - (T(1) - d) * (p.delta + p.lam) * B;
  const T k = (p.delta - p.kappa * (p.delta + p.lam) * d) * B;
  const T cmin = p.cmf * y;
  T Lq;
  const bool capped = issue_cap(q, p, Lq);
  if (mode == 0) {
    if (capped && k > Lq) { H = NEGH<T>(); S = 0; c = cmin; return; }
    T cc = caut + q * k;
    if (cc < cmin) { H = NEGH<T>(); S = 0; c = cmin; return; }
    H = u_fn(cc, p.nu); S = 0; c = cc; return;
  }
  T S_lo = (cmin - caut) / q - k;
  if (S_lo < -p.Scap) S_lo = -p.Scap;
  T S_hi = p.Scap;
  if (capped && S_hi > Lq - k) S_hi = Lq - k;
  if (mode == 1) { if (S_lo < T(0)) S_lo = T(0); }
  else if (mode == -1) { if (S_hi > T(0)) S_hi = T(0); }
  if (S_lo > S_hi) { H = NEGH<T>(); S = 0; c = cmin; return; }
  T Sstar;
  if (VB < T(0)) {
    T cstar = (p.nu == T(2)) ? sqrt(q / (-VB)) : pow(-VB / q, -T(1) / p.nu);
    Sstar = (cstar - caut) / q - k;
  } else {
    Sstar = S_hi;
  }
  T Sv = Sstar;
  if (Sv < S_lo) Sv = S_lo; else if (Sv > S_hi) Sv = S_hi;
  T cc = caut + q * (Sv + k);
  H = u_fn(cc, p.nu) + Sv * VB; S = Sv; c = cc;
}

template <typename T> HD bool finiteH(T H) { return H > T(-1.0e29) && H < T(1.0e30) && H == H; }

// maximize over d in [0,1] and S (pdmodel.best_node)
template <typename T>
HD void best_node(T VB, T q, T B, T y, T Fz, const Par<T>& p, int mode, T& Hb, T& Sb, T& cb, T& db) {
  best_S(T(0), VB, q, B, y, Fz, p, mode, Hb, Sb, cb);
  if (!(Hb == Hb) || Hb > T(1e30)) Hb = NEGH<T>();
  db = T(0);
  T H, S, c;
  if (p.closed_d && p.g1 > T(1) && B > T(0)) {
    T rhs0 = (T(1) - p.kappa * q) * (p.delta + p.lam) * B / (p.g0 * p.g1 * Fz * y);
    if (rhs0 > T(0)) {
      T dc = pow(rhs0, T(1) / (p.g1 - T(1)));
      if (dc < T(1)) {
        best_S(dc, VB, q, B, y, Fz, p, mode, H, S, c);
        if (H > T(-1.0e29) && c > p.cmf * y * (T(1) + T(1.0e-10))) {
          T Lq;
          bool capped = issue_cap(q, p, Lq);
          T kd = (p.delta - p.kappa * (p.delta + p.lam) * dc) * B;
          if ((!capped) || S < Lq - kd - T(1.0e-12)) {
            if (H > Hb) { Hb = H; Sb = S; cb = c; db = dc; }
            return;
          }
        }
      }
    }
  }
  const T INVPHI = T(0.6180339887498949);
  T a = T(1.0e-9), b = T(1);
  T x1 = b - INVPHI * (b - a), x2 = a + INVPHI * (b - a);
  T f1, f2, s_, c_;
  best_S(x1, VB, q, B, y, Fz, p, mode, f1, s_, c_);
  best_S(x2, VB, q, B, y, Fz, p, mode, f2, s_, c_);
  for (int it = 0; it < p.n_golden; ++it) {
    if (f1 <= f2) {
      a = x1; x1 = x2; f1 = f2; x2 = a + INVPHI * (b - a);
      best_S(x2, VB, q, B, y, Fz, p, mode, f2, s_, c_);
    } else {
      b = x2; x2 = x1; f2 = f1; x1 = b - INVPHI * (b - a);
      best_S(x1, VB, q, B, y, Fz, p, mode, f1, s_, c_);
    }
  }
  T dg = T(0.5) * (a + b);
  best_S(dg, VB, q, B, y, Fz, p, mode, H, S, c);
  if (H > Hb) { Hb = H; Sb = S; cb = c; db = dg; }
  best_S(T(1), VB, q, B, y, Fz, p, mode, H, S, c);
  if (H > Hb) { Hb = H; Sb = S; cb = c; db = T(1); }
  if (p.g1 > T(1) && B > T(0)) {
    T rhs = (T(1) - p.kappa * q) * (p.delta + p.lam) * B / (p.g0 * p.g1 * Fz * y);
    if (rhs > T(0)) {
      T di = pow(rhs, T(1) / (p.g1 - T(1)));
      if (di > T(1)) di = T(1);
      best_S(di, VB, q, B, y, Fz, p, mode, H, S, c);
      if (H > Hb) { Hb = H; Sb = S; cb = c; db = di; }
    }
  }
}

// upwind policy at node (i,j) with branch holding (pdmodel.policies). W = V - Vref (differences only).
// Returns which in {1,-1,0,9}; writes S, c, d, u.
template <typename T>
HD int policy_node(int i, int j, int nB, int nz, const T* W, const T* q, const T* Bg, const T* yg, const T* Fz,
                   const T* hf, const T* hb, const Par<T>& p, int wprev, T tau, T& oS, T& oc, T& od, T& ou) {
  const int k = i * nz + j;
  const T B = Bg[i], y = yg[j], F = Fz[j], qq = q[k];
  T bestH = T(-2.0e30), bS = 0, bc = 1, bd = 0;
  int bw = 0;
  const int wp = (tau > T(0)) ? wprev : 9;
  T pH = T(-2.0e30), pS = 0, pc = 1, pd = 0;
  T H, S_, c_, d_;
  if (i < nB - 1) {
    T VBf = (W[k + nz] - W[k]) / hf[i];
    best_node(VBf, qq, B, y, F, p, 1, H, S_, c_, d_);
    if (H > bestH) { bestH = H; bS = S_; bc = c_; bd = d_; bw = 1; }
    if (wp == 1) { pH = H; pS = S_; pc = c_; pd = d_; }
  }
  if (i > 0) {
    T VBb = (W[k] - W[k - nz]) / hb[i];
    best_node(VBb, qq, B, y, F, p, -1, H, S_, c_, d_);
    if (H > bestH) { bestH = H; bS = S_; bc = c_; bd = d_; bw = -1; }
    if (wp == -1) { pH = H; pS = S_; pc = c_; pd = d_; }
  }
  best_node(T(0), qq, B, y, F, p, 0, H, S_, c_, d_);
  if (H > bestH) { bestH = H; bS = S_; bc = c_; bd = d_; bw = 0; }
  if (wp == 0) { pH = H; pS = S_; pc = c_; pd = d_; }
  if (wp != bw && pH > T(-1.0e29) && pH >= bestH - tau) { bestH = pH; bS = pS; bc = pc; bd = pd; bw = wp; }
  if (bestH < T(-1.0e29) || !(bestH == bestH)) {
    bS = 0; bd = 1; bc = phi_fn(T(1), F, p) * y; bw = 9;
  }
  oS = bS; oc = bc; od = bd; ou = u_fn(bc, p.nu);
  return bw;
}

// ---------------------------------------------------------------- counter-based RNG (Philox4x32-10)
struct U4 { uint32_t x, y, z, w; };
HD uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hi) {
  uint64_t p = (uint64_t)a * (uint64_t)b; *hi = (uint32_t)(p >> 32); return (uint32_t)p;
}
HD U4 philox(U4 c, uint32_t k0, uint32_t k1) {
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, hi1;
    uint32_t lo0 = mulhilo32(0xD2511F53u, c.x, &hi0);
    uint32_t lo1 = mulhilo32(0xCD9E8D57u, c.z, &hi1);
    U4 n = {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    c = n; k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  return c;
}
// uniform in (0,1]
template <typename T> HD T u01(uint32_t v) { return (T(v) + T(1)) * T(2.3283064365386963e-10); }
