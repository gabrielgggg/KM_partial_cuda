// gpu_backend.cuh -- the GPU back end. One thread per node for the policies and residuals; the two linear
// systems of a step are solved concurrently, one warp each, by line Gauss-Seidel in which every debt row is
// solved exactly along z with a warp-parallel Thomas algorithm (affine scans across lanes); one thread per
// path segment for the simulation; Crank-Nicolson impulse responses with zebra line relaxation over rows.
#pragma once
#include "shared.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace gpu {

static void warmup() { CK(cudaFree(0)); }

template <typename T> struct DG {
  T *B, *y, *Fz, *hf, *hb, *zup, *zdn, *z;
};
template <typename T> static T* up(const std::vector<T>& v) {
  T* p; CK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(1, v.size())));
  if (!v.empty()) CK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  return p;
}
template <typename T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(1, n))); return p; }
template <typename T> static void down(std::vector<T>& v, const T* p, size_t n) { v.resize(n); CK(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost)); }
template <typename T> static DG<T> upgrid(const Setup& s) {
  DG<T> g;
  g.B = up(castv<T>(s.B)); g.y = up(castv<T>(s.y)); g.Fz = up(castv<T>(s.Fz)); g.hf = up(castv<T>(s.hf));
  g.hb = up(castv<T>(s.hb)); g.zup = up(castv<T>(s.zup)); g.zdn = up(castv<T>(s.zdn)); g.z = up(castv<T>(s.z));
  return g;
}

__device__ __forceinline__ void atomicMaxPos(float* a, float v) { atomicMax((unsigned int*)a, __float_as_uint(v)); }
__device__ __forceinline__ void atomicMaxPos(double* a, double v) { atomicMax((unsigned long long*)a, (unsigned long long)__double_as_longlong(v)); }
template <typename T> __device__ __forceinline__ T warpmax(T v) {
  for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <typename T> __device__ __forceinline__ T warpsum(T v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T>
__global__ void k_policy(int nB, int nz, const T* __restrict__ W, const T* __restrict__ q, DG<T> g, Par<T> p,
                         int8_t* which, T* tau, uint32_t* mask, uint8_t* frozen, T* S, T* c, T* d, T* u,
                         int first, int count, int flips, uint32_t wmask) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nB * nz) return;
  const int i = k / nz, j = k - i * nz;
  const int wprev = which[k];
  const int wp = first ? 9 : wprev;
  const T tk = first ? T(0) : tau[k];
  T S_, c_, d_, u_;
  int w = policy_node<T>(i, j, nB, nz, W, q, g.B, g.y, g.Fz, g.hf, g.hb, p, wp, tk, S_, c_, d_, u_);
  S[k] = S_; c[k] = c_; d[k] = d_; u[k] = u_;
  if (count && flips) {
    uint32_t mk = ((mask[k] << 1) | (w != wprev ? 1u : 0u)) & wmask;
    mask[k] = mk;
    if (!frozen[k] && __popc(mk) >= flips) { frozen[k] = 1; tau[k] = T(1e30); }
  }
  which[k] = (int8_t)w;
}

template <typename T>
__global__ void k_resid(int nB, int nz, const T* W, const T* q, const T* S, const T* u, const T* d, DG<T> g, Par<T> p, T Vref, T* out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  T a = 0, b = 0;
  if (k < nB * nz) resid_node<T>(k / nz, k % nz, nB, nz, W, q, S, u, d, g.hf, g.hb, g.zup, g.zdn, p, Vref, a, b);
  a = warpmax(a); b = warpmax(b);
  if ((threadIdx.x & 31) == 0) { atomicMaxPos(&out[0], a); atomicMaxPos(&out[1], b); }
}

template <typename T> struct SysP { T *D, *m, *inv, *cp, *b, *x; };

// one warp per (system, debt row): the lanes form the diagonal, right-hand side and upwind coefficients of the
// row (coalesced), lane 0 runs the Thomas factorization from shared memory, and the lanes store the factors.
// The system-0 warps also store the upwind coefficients on the neighbouring rows (cH on i+1, cL on i-1).
template <typename T>
__global__ void __launch_bounds__(128) k_factor(int nB, int nz, const T* S, DG<T> g, const T* u, const T* d, Par<T> p, T Vref, T idt, SysP<T> sV, SysP<T> sq, T* cH, T* cL) {
  __shared__ T sh[4][4][128];   // per warp: D, inv, cp, m
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * 4 + w;
  if (t >= 2 * nB) return;
  const int sys = t / nB, i = t % nB;
  SysP<T> sy = sys == 0 ? sV : sq;
  const T coupon = p.delta + p.lam;
  const T ihf = (i < nB - 1) ? T(1) / g.hf[i] : T(0), ihb = (i > 0) ? T(1) / g.hb[i] : T(0);
  T* Ds = sh[w][0]; T* Is = sh[w][1]; T* Cs = sh[w][2]; T* Ms = sh[w][3];
  for (int j = lane; j < nz; j += 32) {
    const int k = i * nz + j; const T s = S[k];
    const T ch = s > T(0) ? s * ihf : T(0), cl = s < T(0) ? -s * ihb : T(0);
    if (sys == 0) { cH[k] = ch; cL[k] = cl; }
    T cc, bb;
    if (sys == 0) { cc = p.rho + idt; bb = u[k] - p.rho * Vref + sy.x[k] * idt; }
    else { cc = p.r + p.delta - p.kappa * coupon * d[k] + idt; bb = (T(1) - d[k]) * coupon + sy.x[k] * idt; }
    // the diagonal keeps S/h on the boundary rows as the generator does (S = 0 there in practice)
    const T D = cc + (s > T(0) ? s / g.hf[i] : T(0)) - (s < T(0) ? s / g.hb[i] : T(0)) + g.zup[j] + g.zdn[j];
    sy.D[k] = D; sy.b[k] = bb; Ds[j] = D;
  }
  __syncwarp();
  if (lane == 0) {
    T cpp = 0;
    for (int j = 0; j < nz; ++j) {
      const T den = Ds[j] + (j > 0 ? g.zdn[j] * cpp : T(0));
      const T iv = T(1) / den;
      cpp = -g.zup[j] * iv;
      Is[j] = iv; Cs[j] = cpp; Ms[j] = j > 0 ? g.zdn[j] * iv : T(0);
    }
  }
  __syncwarp();
  for (int j = lane; j < nz; j += 32) { const int k = i * nz + j; sy.inv[k] = Is[j]; sy.cp[k] = Cs[j]; sy.m[k] = Ms[j]; }
}

// starting guess of the sweeps: linear extrapolation of the last two iterates
template <typename T> __global__ void k_extrap(int N, T* W, T* q, T* Wp, T* qp, int first) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  const T w = W[k], qq = q[k];
  if (!first) { W[k] = w + (w - Wp[k]); q[k] = qq + (qq - qp[k]); }
  Wp[k] = w; qp[k] = qq;
}

// line Gauss-Seidel, one warp per system; lane l owns z-nodes l*K .. l*K+K-1 of every row. The loads of the
// next row are issued before the current row is solved (software pipelining).
template <typename T, int K>
__global__ void __launch_bounds__(32) k_sweep(int nB, int nz, const T* __restrict__ cH, const T* __restrict__ cL,
                                              const T* __restrict__ zup, const T* __restrict__ zdn, SysP<T> s0, SysP<T> s1, T tolrel,
                                              int maxpass, int stall, unsigned long long* passes, int* uncon, int clip) {
  SysP<T> sy = blockIdx.x == 0 ? s0 : s1;
  const int lane = threadIdx.x;
  const T* __restrict__ D = sy.D; const T* __restrict__ m = sy.m; const T* __restrict__ inv = sy.inv;
  const T* __restrict__ cp = sy.cp; const T* __restrict__ b = sy.b; T* x = sy.x;
  T bm = 0;
#pragma unroll 4
  for (int i = 0; i < nB; ++i)
#pragma unroll
    for (int e = 0; e < K; ++e) { int j = lane * K + e; if (j < nz) bm = fmax(bm, fabs(b[i * nz + j])); }
  bm = warpmax(bm);
  const T tol = tolrel * fmax(T(1), bm);
  T zu[K], zd[K]; bool val[K];
#pragma unroll
  for (int e = 0; e < K; ++e) { int j = lane * K + e; val[e] = j < nz; zu[e] = val[e] ? zup[j] : T(0); zd[e] = val[e] ? zdn[j] : T(0); }
  int pass = 0; T res = T(1e30);
  bool stalled = false;
  for (pass = 0; pass < maxpass; ++pass) {
    for (int sw = 0; sw < 2; ++sw) {
      T xs[K];
#pragma unroll
      for (int e = 0; e < K; ++e) xs[e] = 0;
      // registers of the next row
      T nb[K], nh[K], nl[K], nm[K], ni[K], nc[K], nx[K];
      auto load = [&](int i) {
        const int io = sw == 0 ? (i < nB - 1 ? i + 1 : i) : (i > 0 ? i - 1 : i);   // neighbour row from the last pass
#pragma unroll
        for (int e = 0; e < K; ++e) {
          const int j = lane * K + e;
          if (val[e]) {
            const int k = i * nz + j;
            nb[e] = b[k]; nh[e] = cH[k]; nl[e] = cL[k]; nm[e] = m[k]; ni[e] = inv[k]; nc[e] = cp[k]; nx[e] = x[io * nz + j];
          } else { nb[e] = 0; nh[e] = 0; nl[e] = 0; nm[e] = 0; ni[e] = 0; nc[e] = 0; nx[e] = 0; }
        }
      };
      load(sw == 0 ? 0 : nB - 1);
      for (int ii = 0; ii < nB; ++ii) {
        const int i = sw == 0 ? ii : nB - 1 - ii;
        T r[K], mm[K], iv[K], cc[K];
#pragma unroll
        for (int e = 0; e < K; ++e) {
          // up sweep: row i-1 is in xs (just solved), row i+1 from memory; down sweep the reverse
          const T xu = sw == 0 ? nx[e] : xs[e];
          const T xd = sw == 0 ? xs[e] : nx[e];
          r[e] = nb[e] + nh[e] * xu + nl[e] * xd;
          mm[e] = nm[e]; iv[e] = ni[e]; cc[e] = nc[e];
        }
        if (ii + 1 < nB) load(sw == 0 ? i + 1 : i - 1);
        T M = 1, E = 0;
#pragma unroll
        for (int e = 0; e < K; ++e) { E = mm[e] * E + iv[e] * r[e]; M = mm[e] * M; }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          T Mo = __shfl_up_sync(0xffffffffu, M, o), Eo = __shfl_up_sync(0xffffffffu, E, o);
          if (lane >= o) { E = M * Eo + E; M = M * Mo; }
        }
        T yv = __shfl_up_sync(0xffffffffu, E, 1);
        if (lane == 0) yv = 0;
        T ys[K];
#pragma unroll
        for (int e = 0; e < K; ++e) { yv = mm[e] * yv + iv[e] * r[e]; ys[e] = yv; }
        M = 1; E = 0;
#pragma unroll
        for (int e = K - 1; e >= 0; --e) { E = -cc[e] * E + ys[e]; M = -cc[e] * M; }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          T Mo = __shfl_down_sync(0xffffffffu, M, o), Eo = __shfl_down_sync(0xffffffffu, E, o);
          if (lane + o < 32) { E = M * Eo + E; M = M * Mo; }
        }
        T xv = __shfl_down_sync(0xffffffffu, E, 1);
        if (lane == 31) xv = 0;
#pragma unroll
        for (int e = K - 1; e >= 0; --e) { xv = ys[e] - cc[e] * xv; xs[e] = xv; }
#pragma unroll
        for (int e = 0; e < K; ++e) if (val[e]) x[i * nz + lane * K + e] = xs[e];
      }
    }
    __syncwarp();
    const T prev = res;
    res = 0;
#pragma unroll 4
    for (int i = 0; i < nB; ++i) {
      const int iu = i < nB - 1 ? i + 1 : i, id = i > 0 ? i - 1 : i;
#pragma unroll
      for (int e = 0; e < K; ++e) {
        const int j = lane * K + e;
        if (val[e]) {
          const int k = i * nz + j;
          T rr = D[k] * x[k] - b[k] - cH[k] * x[iu * nz + j] - cL[k] * x[id * nz + j];
          if (j < nz - 1) rr -= zu[e] * x[k + 1];
          if (j > 0) rr -= zd[e] * x[k - 1];
          res = fmax(res, fabs(rr));
        }
      }
    }
    res = warpmax(res);
    if (res < tol) break;
    if (stall && res > T(0.9) * prev) { stalled = true; break; }
  }
  if (lane == 0) { atomicAdd(passes, (unsigned long long)(pass < maxpass ? pass + 1 : maxpass)); if (pass >= maxpass || stalled) atomicAdd(uncon, 1); }
  if (clip && blockIdx.x == 1) {
    for (int k = lane; k < nB * nz; k += 32) x[k] = fmin(T(1), fmax(T(1e-12), x[k]));
  }
}

// Block line Gauss-Seidel for larger grids: the debt rows of each system are cut into P chunks, one warp per
// chunk, all chunks of both systems swept at once (a cooperative grid). Inside a chunk the sweep is Gauss-Seidel
// as in k_sweep; across a chunk boundary the neighbouring row is taken from the previous half-sweep (Jacobi).
// Upwind coupling decays geometrically along a chunk, so the extra passes are few while the sweep is P times
// shorter. The stopping rule is that of k_sweep: the maximum residual of the system below tolrel*max(1,|b|).
template <typename T, int K>
__global__ void __launch_bounds__(32) k_sweepc(int nB, int nz, int P, const T* __restrict__ cH, const T* __restrict__ cL,
                                               const T* __restrict__ zup, const T* __restrict__ zdn, SysP<T> s0, SysP<T> s1, T tolrel,
                                               int maxpass, int stall, T* bnd, T* red, unsigned long long* passes, int* uncon, int clip) {
  cg::grid_group grid = cg::this_grid();
  const int sys = blockIdx.x / P, c = blockIdx.x % P, lane = threadIdx.x;
  SysP<T> sy = sys == 0 ? s0 : s1;
  const int i0 = (int)((long)c * nB / P), i1 = (int)((long)(c + 1) * nB / P);
  const T* __restrict__ D = sy.D; const T* __restrict__ m = sy.m; const T* __restrict__ inv = sy.inv;
  const T* __restrict__ cp = sy.cp; const T* __restrict__ b = sy.b; T* x = sy.x;
  // bnd[(par, sys, c, side)][nz]: first (side 0) and last (side 1) row of every chunk after the last half-sweep
  auto B_ = [&](int par, int s, int cc, int side) { return bnd + ((((size_t)par * 2 + s) * P + cc) * 2 + side) * nz; };
  T* R_ = red;   // red[(slot, sys, c)]: slot 0 max|b|, slots 1, 2 residual of the pass (alternating)
  bool val[K]; T zu[K], zd[K];
#pragma unroll
  for (int e = 0; e < K; ++e) { int j = lane * K + e; val[e] = j < nz; zu[e] = val[e] ? zup[j] : T(0); zd[e] = val[e] ? zdn[j] : T(0); }
  T bm = 0;
  for (int i = i0; i < i1; ++i)
#pragma unroll
    for (int e = 0; e < K; ++e) if (val[e]) bm = fmax(bm, fabs(b[i * nz + lane * K + e]));
  bm = warpmax(bm);
  if (lane == 0) R_[(0 * 2 + sys) * P + c] = bm;
  auto snap = [&](int par) {
#pragma unroll
    for (int e = 0; e < K; ++e) if (val[e]) { const int j = lane * K + e; B_(par, sys, c, 0)[j] = x[i0 * nz + j]; B_(par, sys, c, 1)[j] = x[(i1 - 1) * nz + j]; }
  };
  snap(0);
  grid.sync();
  T bmax = 0;
  for (int cc = 0; cc < P; ++cc) bmax = fmax(bmax, R_[(0 * 2 + sys) * P + cc]);
  const T tol = tolrel * fmax(T(1), bmax);
  int par = 0, pass, npass = 0;
  bool done = false, stalled = false;
  T res = T(1e30);
  for (pass = 0; pass < maxpass; ++pass) {
    for (int sw = 0; sw < 2; ++sw) {
      if (!done) {
        T xs[K];
        // row entering the chunk from the already swept side: the neighbouring chunk's boundary row (previous half-sweep)
#pragma unroll
        for (int e = 0; e < K; ++e) {
          const int j = lane * K + e;
          if (sw == 0) xs[e] = (c > 0 && val[e]) ? B_(par, sys, c - 1, 1)[j] : T(0);
          else xs[e] = (c < P - 1 && val[e]) ? B_(par, sys, c + 1, 0)[j] : T(0);
        }
        T nb[K], nh[K], nl[K], nm[K], ni[K], nc[K], nx[K];
        auto load = [&](int i) {
          const int io = sw == 0 ? (i < nB - 1 ? i + 1 : i) : (i > 0 ? i - 1 : i);
          const bool outside = sw == 0 ? (io >= i1) : (io < i0);
#pragma unroll
          for (int e = 0; e < K; ++e) {
            const int j = lane * K + e;
            if (val[e]) {
              const int k = i * nz + j;
              nb[e] = b[k]; nh[e] = cH[k]; nl[e] = cL[k]; nm[e] = m[k]; ni[e] = inv[k]; nc[e] = cp[k];
              nx[e] = outside ? (sw == 0 ? B_(par, sys, c + 1, 0)[j] : B_(par, sys, c - 1, 1)[j]) : x[io * nz + j];
            } else { nb[e] = 0; nh[e] = 0; nl[e] = 0; nm[e] = 0; ni[e] = 0; nc[e] = 0; nx[e] = 0; }
          }
        };
        const int n = i1 - i0;
        load(sw == 0 ? i0 : i1 - 1);
        for (int ii = 0; ii < n; ++ii) {
          const int i = sw == 0 ? i0 + ii : i1 - 1 - ii;
          T r[K], mm[K], iv[K], cc_[K];
#pragma unroll
          for (int e = 0; e < K; ++e) {
            const T xu = sw == 0 ? nx[e] : xs[e];
            const T xd = sw == 0 ? xs[e] : nx[e];
            r[e] = nb[e] + nh[e] * xu + nl[e] * xd;
            mm[e] = nm[e]; iv[e] = ni[e]; cc_[e] = nc[e];
          }
          if (ii + 1 < n) load(sw == 0 ? i + 1 : i - 1);
          T M = 1, E = 0;
#pragma unroll
          for (int e = 0; e < K; ++e) { E = mm[e] * E + iv[e] * r[e]; M = mm[e] * M; }
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            T Mo = __shfl_up_sync(0xffffffffu, M, o), Eo = __shfl_up_sync(0xffffffffu, E, o);
            if (lane >= o) { E = M * Eo + E; M = M * Mo; }
          }
          T yv = __shfl_up_sync(0xffffffffu, E, 1);
          if (lane == 0) yv = 0;
          T ys[K];
#pragma unroll
          for (int e = 0; e < K; ++e) { yv = mm[e] * yv + iv[e] * r[e]; ys[e] = yv; }
          M = 1; E = 0;
#pragma unroll
          for (int e = K - 1; e >= 0; --e) { E = -cc_[e] * E + ys[e]; M = -cc_[e] * M; }
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            T Mo = __shfl_down_sync(0xffffffffu, M, o), Eo = __shfl_down_sync(0xffffffffu, E, o);
            if (lane + o < 32) { E = M * Eo + E; M = M * Mo; }
          }
          T xv = __shfl_down_sync(0xffffffffu, E, 1);
          if (lane == 31) xv = 0;
#pragma unroll
          for (int e = K - 1; e >= 0; --e) { xv = ys[e] - cc_[e] * xv; xs[e] = xv; }
#pragma unroll
          for (int e = 0; e < K; ++e) if (val[e]) x[i * nz + lane * K + e] = xs[e];
        }
        __syncwarp();
        snap(par ^ 1);
      }
      par ^= 1;
      grid.sync();
      if (done) {   // keep the boundary buffers of a finished system current for the parity flip
#pragma unroll
        for (int e = 0; e < K; ++e) if (val[e]) { const int j = lane * K + e; B_(par, sys, c, 0)[j] = B_(par ^ 1, sys, c, 0)[j]; B_(par, sys, c, 1)[j] = B_(par ^ 1, sys, c, 1)[j]; }
      }
    }
    const int slot = 1 + (pass & 1);
    if (!done) {
      ++npass;
      T rl = 0;
      for (int i = i0; i < i1; ++i) {
        const int iu = i < nB - 1 ? i + 1 : i, id = i > 0 ? i - 1 : i;
#pragma unroll
        for (int e = 0; e < K; ++e) {
          const int j = lane * K + e;
          if (val[e]) {
            const int k = i * nz + j;
            T rr = D[k] * x[k] - b[k] - cH[k] * x[iu * nz + j] - cL[k] * x[id * nz + j];
            if (j < nz - 1) rr -= zu[e] * x[k + 1];
            if (j > 0) rr -= zd[e] * x[k - 1];
            rl = fmax(rl, fabs(rr));
          }
        }
      }
      rl = warpmax(rl);
      if (lane == 0) R_[(slot * 2 + sys) * P + c] = rl;
    }
    grid.sync();
    bool fin[2];
    for (int s2 = 0; s2 < 2; ++s2) fin[s2] = false;
    // every block evaluates both systems' state so that all leave the loop together
    {
      T r2 = 0;
      for (int cc = 0; cc < P; ++cc) r2 = fmax(r2, R_[(slot * 2 + sys) * P + cc]);
      if (!done) {
        const T prev = res; res = r2;
        if (res < tol) done = true;
        else if (stall && res > T(0.9) * prev) { done = true; stalled = true; }
      }
    }
    // both systems done? each block's 'done' is identical within its system; combine through red slot 0 area
    if (lane == 0 && c == 0) R_[(3 * 2 + sys) * P] = done ? T(1) : T(0);
    grid.sync();
    fin[0] = R_[(3 * 2 + 0) * P] > T(0.5); fin[1] = R_[(3 * 2 + 1) * P] > T(0.5);
    if (fin[0] && fin[1]) break;
    grid.sync();   // the flags are read before they are rewritten in the next pass
  }
  if (lane == 0 && c == 0) { atomicAdd(passes, (unsigned long long)npass); if (!done || stalled) atomicAdd(uncon, 1); }
  if (clip && sys == 1)
    for (int k = i0 * nz + lane; k < i1 * nz; k += 32) x[k] = fmin(T(1), fmax(T(1e-12), x[k]));
}

template <typename T>
static void solve(const Setup& s, const Opt& o, const std::vector<double>& V0, const std::vector<double>& q0, Sol<T>& sol) {
  const int nB = s.nB, nz = s.nz, N = s.N;
  if (nz > 128) { fprintf(stderr, "nz > 128 not supported by the warp solver\n"); exit(1); }
  DG<T> g = upgrid<T>(s);
  Par<T> p = castp<T>(s.p);
  double vr = 0; for (double v : V0) vr += v; vr /= N;
  sol.Vref = vr; const T Vref = (T)vr;
  std::vector<T> W(N), q(N);
  for (int k = 0; k < N; ++k) { W[k] = (T)(V0[k] - vr); q[k] = (T)q0[k]; }
  T *dW = up(W), *dq = up(q), *dS = dalloc<T>(N), *dc = dalloc<T>(N), *dd = dalloc<T>(N), *du = dalloc<T>(N);
  T* dtau = up(std::vector<T>(N, (T)o.sticky));
  int8_t* dwhich = up(std::vector<int8_t>(N, 9));
  uint32_t* dmask = up(std::vector<uint32_t>(N, 0));
  uint8_t* dfro = up(std::vector<uint8_t>(N, 0));
  SysP<T> sV{dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dW};
  SysP<T> sq{dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dalloc<T>(N), dq};
  T* dres = dalloc<T>(2);
  T *dcH = dalloc<T>(N), *dcL = dalloc<T>(N), *dWp = dalloc<T>(N), *dqp = dalloc<T>(N);
  unsigned long long* dpass; int* duncon;
  CK(cudaMalloc(&dpass, 8)); CK(cudaMalloc(&duncon, 4)); CK(cudaMemset(dpass, 0, 8)); CK(cudaMemset(duncon, 0, 4));
  T* hres; CK(cudaMallocHost(&hres, 2 * sizeof(T)));
  // chunks of the block sweep: about o.chunk_rows rows each (0: one warp per system, k_sweep)
  const int crows = o.chunk_rows >= 0 ? o.chunk_rows : 16;
  int P = crows > 0 ? std::max(1, nB / crows) : 1;
  {  // all blocks of the cooperative sweep must be resident at once
    int dev, nsm, per = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    const int K_ = (nz + 31) / 32;
    const void* fn = K_ == 1 ? (const void*)k_sweepc<T, 1> : K_ == 2 ? (const void*)k_sweepc<T, 2> : K_ == 3 ? (const void*)k_sweepc<T, 3> : (const void*)k_sweepc<T, 4>;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, 32, 0));
    P = std::max(1, std::min(P, per * nsm / 2));
  }
  T* dbnd = dalloc<T>((size_t)2 * 2 * P * 2 * nz); T* dred = dalloc<T>((size_t)4 * 2 * P);
  const uint32_t wmask = (o.freeze_window >= 32) ? 0xFFFFFFFFu : ((1u << o.freeze_window) - 1u);
  const T stol = (T)(sizeof(T) == 4 ? std::max(o.sweep_tol, 1e-6) : o.sweep_tol);
  const int K = (nz + 31) / 32;
  const int TB = 128, NBk = (N + TB - 1) / TB;
  double dt = o.dt0, dta = 0, rV = 0, rq = 0;
  int it; bool conv = false;
  double best_m = 1e300, best_q = 1e300; int best_it = 0;
  for (it = 1; it <= o.maxit; ++it) {
    const int first = it == 1, count = !first && dt >= o.dt_max;
    k_policy<T><<<NBk, TB>>>(nB, nz, dW, dq, g, p, dwhich, dtau, dmask, dfro, dS, dc, dd, du, first, count, o.freeze_flips, wmask);
    CK(cudaMemsetAsync(dres, 0, 2 * sizeof(T)));
    k_resid<T><<<NBk, TB>>>(nB, nz, dW, dq, dS, du, dd, g, p, Vref, dres);
    CK(cudaMemcpy(hres, dres, 2 * sizeof(T), cudaMemcpyDeviceToHost));
    rV = hres[0]; rq = hres[1];
    if (o.verbose > 1 && (it % 50 == 0 || it == 1)) printf("step %5d dt %.4f resV %.3e resq %.3e\n", it, dt, rV, rq);
    if (rV < o.tolV && rq < o.tolq) { conv = true; break; }
    // single precision: tolerances scaled to the precision (relative to |V| and to the unit price), and a stop
    // once the residuals stop improving (they sit at the rounding floor of the scheme)
    if (sizeof(T) == 4 && rV < o.tol32 * 1.1920929e-7 * std::fabs(sol.Vref) && rq < o.tol32 * 1.1920929e-7) { conv = true; break; }
    if (sizeof(T) == 4 && dt >= o.dt_max) {
      // a new low of either residual restarts the window
      if (rV < 0.98 * best_m) { best_m = rV; best_it = it; }
      if (rq < 0.98 * best_q) { best_q = rq; best_it = it; }
      if (it - best_it >= o.stall_window) { sol.stalled = true; break; }
    }
    dt = std::min(dt * o.dt_grow, o.dt_max);
    if (o.accel_on > 0 && rV < o.accel_on && rq < o.accel_on) { dta = std::max(dta, dt); dta = std::min(dta * o.accel_grow, o.accel_dtmax); dt = dta; }
    k_factor<T><<<(2 * nB + 3) / 4, 128>>>(nB, nz, dS, g, du, dd, p, Vref, (T)(1.0 / dt), sV, sq, dcH, dcL);
    if (o.extrap) k_extrap<T><<<NBk, TB>>>(N, dW, dq, dWp, dqp, it == 1);
    const int stall = sizeof(T) == 4;
    if (P > 1) {
      int maxp = o.maxpass, stl = stall, clp = 1;
      T st_ = stol;
      void* args[] = {(void*)&nB, (void*)&nz, (void*)&P, (void*)&dcH, (void*)&dcL, (void*)&g.zup, (void*)&g.zdn, (void*)&sV, (void*)&sq,
                      (void*)&st_, (void*)&maxp, (void*)&stl, (void*)&dbnd, (void*)&dred, (void*)&dpass, (void*)&duncon, (void*)&clp};
      void* fn = K == 1 ? (void*)k_sweepc<T, 1> : K == 2 ? (void*)k_sweepc<T, 2> : K == 3 ? (void*)k_sweepc<T, 3> : (void*)k_sweepc<T, 4>;
      CK(cudaLaunchCooperativeKernel(fn, dim3(2 * P), dim3(32), args, 0, 0));
    } else
    switch (K) {
      case 1: k_sweep<T, 1><<<2, 32>>>(nB, nz, dcH, dcL, g.zup, g.zdn, sV, sq, stol, o.maxpass, stall, dpass, duncon, 1); break;
      case 2: k_sweep<T, 2><<<2, 32>>>(nB, nz, dcH, dcL, g.zup, g.zdn, sV, sq, stol, o.maxpass, stall, dpass, duncon, 1); break;
      case 3: k_sweep<T, 3><<<2, 32>>>(nB, nz, dcH, dcL, g.zup, g.zdn, sV, sq, stol, o.maxpass, stall, dpass, duncon, 1); break;
      default: k_sweep<T, 4><<<2, 32>>>(nB, nz, dcH, dcL, g.zup, g.zdn, sV, sq, stol, o.maxpass, stall, dpass, duncon, 1); break;
    }
  }
  k_policy<T><<<NBk, TB>>>(nB, nz, dW, dq, g, p, dwhich, dtau, dmask, dfro, dS, dc, dd, du, 0, 0, o.freeze_flips, wmask);
  CK(cudaDeviceSynchronize());
  down(sol.W, dW, N); down(sol.q, dq, N); down(sol.S, dS, N); down(sol.c, dc, N); down(sol.d, dd, N); down(sol.u, du, N);
  down(sol.which, dwhich, N); down(sol.frozen, dfro, N);
  unsigned long long hp; int hu;
  CK(cudaMemcpy(&hp, dpass, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hu, duncon, 4, cudaMemcpyDeviceToHost));
  sol.passes = (long)hp; sol.unconverged_solves = hu;
  sol.iters = std::min(it, o.maxit); sol.resV = rV; sol.resq = rq; sol.converged = conv;
  sol.n_frozen = 0; for (auto f : sol.frozen) sol.n_frozen += f;
  for (T* ptr : {dcH, dcL, dWp, dqp, dW, dq, dS, dc, dd, du, dtau, sV.D, sV.m, sV.inv, sV.cp, sV.b, sq.D, sq.m, sq.inv, sq.cp, sq.b, dres, dbnd, dred}) cudaFree(ptr);
  cudaFree(dwhich); cudaFree(dmask); cudaFree(dfro); cudaFree(dpass); cudaFree(duncon); cudaFreeHost(hres);
}

template <typename T> __global__ void k_sim(int n, SimArgs<T> a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) sim_segment<T>(t, a);
}

template <typename T>
static void simulate(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& P, int& Y, int& L) {
  DG<T> g = upgrid<T>(s);
  std::vector<T> cdf(s.N); double acc = 0, tot = 0;
  for (auto v : sol.g) tot += v;
  for (int k = 0; k < s.N; ++k) { acc += sol.g[k] / tot; cdf[k] = (T)acc; }
  SimArgs<T> a;
  a.S = up(sol.S); a.d = up(sol.d); a.c = up(sol.c); a.q = up(sol.q); a.Bg = g.B; a.zg = g.z; a.cdf = up(cdf);
  a.nB = s.nB; a.nz = s.nz; a.N = s.N; a.mu = (T)s.mu; a.sigma = (T)s.sigma; a.delta_lam = (T)(s.delta + s.lam); a.r = (T)s.r; a.delta = (T)s.delta;
  a.g0 = (T)s.p.g0; a.g1 = (T)s.p.g1; a.g2 = (T)s.p.g2; a.zt = (T)s.p.zt; a.Fmin = (T)s.p.Fmin;
  a.P = o.n_paths; a.spy = o.spy; a.seed = o.seed; a.Y = o.years;
  if (o.sim_layout == 0) { a.nseg = 1; a.L = o.years; a.burn = o.burn_paper; a.paper_start = 1; a.B0 = (T)(0.25 * (s.B[0] + s.B[s.nB - 1])); a.z0 = 0; }
  else { a.nseg = o.years / o.seg_years; a.L = o.seg_years; a.burn = o.burn_seg; a.paper_start = 0; a.B0 = 0; a.z0 = 0; a.Y = a.nseg * a.L; }
  const size_t n = (size_t)8 * a.P * a.Y;
  a.out = dalloc<T>(n);
  const int nthr = a.P * a.nseg;
  k_sim<T><<<(nthr + 63) / 64, 64>>>(nthr, a);
  CK(cudaGetLastError());
  std::vector<T> buf; down(buf, a.out, n);
  out.assign(buf.begin(), buf.end());
  P = a.P; Y = a.Y; L = a.L;
}

template <typename T>
static void irf(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& nrec);

}  // namespace gpu
