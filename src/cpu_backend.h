// cpu_backend.h -- the CPU back end: OpenMP over nodes for the policies, residuals, simulation and impulse
// responses; the two linear systems of a step are solved concurrently (one thread each) by line Gauss-Seidel.
#pragma once
#include "shared.h"

namespace cpu {

template <typename T> struct G {
  std::vector<T> B, y, Fz, hf, hb, zup, zdn, z;
  explicit G(const Setup& s) : B(castv<T>(s.B)), y(castv<T>(s.y)), Fz(castv<T>(s.Fz)), hf(castv<T>(s.hf)),
                               hb(castv<T>(s.hb)), zup(castv<T>(s.zup)), zdn(castv<T>(s.zdn)), z(castv<T>(s.z)) {}
};

// line Gauss-Seidel for [diag(cc) - A(S)] x = b: exact tridiagonal solves along z for each debt row, rows upward
// then downward, until the largest residual is below tol (pdmodel.line_sweep). Factors are computed once.
template <typename T>
static int sweep(int nB, int nz, const T* cH, const T* cL, const G<T>& g, const T* cc, const T* b, T* x, T tolrel, int maxpass,
                 T& resout, std::vector<T>& D, std::vector<T>& m, std::vector<T>& inv, std::vector<T>& cp) {
  const int N = nB * nz;
  T bmax = 0;
  for (int k = 0; k < N; ++k) bmax = std::max(bmax, (T)fabs(b[k]));
  const T tol = tolrel * std::max(T(1), bmax);
  std::vector<T> lo(nz), up(nz);
  for (int j = 0; j < nz; ++j) { lo[j] = g.zdn[j]; up[j] = g.zup[j]; }
  for (int i = 0; i < nB; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int k = i * nz + j;
      D[k] = cc[k] + cH[k] + cL[k] + g.zup[j] + g.zdn[j];
    }
    thomas_factor(nz, &D[i * nz], lo.data(), up.data(), &m[i * nz], &inv[i * nz], &cp[i * nz]);
  }
  std::vector<T> yb(nz);
  T res = T(1e30);
  for (int it = 0; it < maxpass; ++it) {
    for (int sw = 0; sw < 2; ++sw) {
      for (int ii = 0; ii < nB; ++ii) {
        const int i = sw == 0 ? ii : nB - 1 - ii;
        const int k0 = i * nz;
        T yp = 0;
        const T* xu = (i < nB - 1) ? x + k0 + nz : x + k0;   // cH = 0 on the top row, cL = 0 on the bottom row
        const T* xd = (i > 0) ? x + k0 - nz : x + k0;
        for (int j = 0; j < nz; ++j) {
          const int k = k0 + j;
          const T r = b[k] + cH[k] * xu[j] + cL[k] * xd[j];
          yp = inv[k] * r + m[k] * yp; yb[j] = yp;
        }
        T xn = 0;
        for (int j = nz - 1; j >= 0; --j) { xn = yb[j] - cp[k0 + j] * xn; x[k0 + j] = xn; }
      }
    }
    T prev = res;
    res = 0;
    for (int i = 0; i < nB; ++i) {
      const int k0 = i * nz;
      const T* xu = (i < nB - 1) ? x + k0 + nz : x + k0;
      const T* xd = (i > 0) ? x + k0 - nz : x + k0;
      for (int j = 0; j < nz; ++j) {
        const int k = k0 + j;
        T r = D[k] * x[k] - b[k] - cH[k] * xu[j] - cL[k] * xd[j];
        if (j < nz - 1) r -= g.zup[j] * x[k + 1];
        if (j > 0) r -= g.zdn[j] * x[k - 1];
        res = std::max(res, (T)fabs(r));
      }
    }
    if (res < tol) { resout = res; return it + 1; }
    if (sizeof(T) == 4 && res > T(0.9) * prev) { resout = res; return 1000 + it + 1; }   // no progress: residual at its rounding floor
  }
  resout = res;
  return -maxpass;
}

// Block line Gauss-Seidel on all cores (as gpu::k_sweepc): the rows of both systems are cut into P chunks each,
// the 2P chunks are swept in parallel, Gauss-Seidel inside a chunk and the previous half-sweep's boundary row
// across chunk boundaries. Returns the passes of each system in n[2] (negative: not converged; +1000: stalled).
template <typename T>
static void sweepc(int nB, int nz, int P, const T* cH, const T* cL, const G<T>& g, const T* cc0, const T* b0, T* x0,
                   const T* cc1, const T* b1, T* x1, T tolrel, int maxpass, int n[2], std::vector<T>* D, std::vector<T>* m,
                   std::vector<T>* inv, std::vector<T>* cp, std::vector<T>& bnd) {
  const int N = nB * nz, NT = 2 * P;
  const T* ccs[2] = {cc0, cc1}; const T* bs[2] = {b0, b1}; T* xs[2] = {x0, x1};
  bnd.resize((size_t)2 * NT * nz);
  std::vector<T> resc(NT, 0), bmc(NT, 0);
  T tol[2], res[2] = {T(1e30), T(1e30)};
  bool done[2] = {false, false};
  int np[2] = {0, 0}, flag[2] = {0, 0};
  auto lo_i = [&](int c) { return (int)((long)c * nB / P); };
#pragma omp parallel
  {
    std::vector<T> yb(nz);
#pragma omp for schedule(static)
    for (int r = 0; r < 2 * nB; ++r) {
      const int s = r / nB, i = r % nB;
      for (int j = 0; j < nz; ++j) { const int k = i * nz + j; D[s][k] = ccs[s][k] + cH[k] + cL[k] + g.zup[j] + g.zdn[j]; }
      thomas_factor(nz, &D[s][i * nz], g.zdn.data(), g.zup.data(), &m[s][i * nz], &inv[s][i * nz], &cp[s][i * nz]);
    }
#pragma omp for schedule(static)
    for (int t = 0; t < NT; ++t) {
      const int s = t / P, c = t % P; T bm = 0;
      for (int k = lo_i(c) * nz; k < lo_i(c + 1) * nz; ++k) bm = std::max(bm, (T)fabs(bs[s][k]));
      bmc[t] = bm;
    }
#pragma omp single
    for (int s = 0; s < 2; ++s) { T bm = 0; for (int c = 0; c < P; ++c) bm = std::max(bm, bmc[s * P + c]); tol[s] = tolrel * std::max(T(1), bm); }
    for (int pass = 0; pass < maxpass; ++pass) {
      for (int sw = 0; sw < 2; ++sw) {
        // boundary rows of every chunk as left by the previous half-sweep
#pragma omp for schedule(static)
        for (int t = 0; t < NT; ++t) {
          const int s = t / P, c = t % P;
          std::copy(xs[s] + lo_i(c) * nz, xs[s] + lo_i(c) * nz + nz, &bnd[((size_t)t * 2 + 0) * nz]);
          std::copy(xs[s] + (lo_i(c + 1) - 1) * nz, xs[s] + lo_i(c + 1) * nz, &bnd[((size_t)t * 2 + 1) * nz]);
        }
#pragma omp for schedule(static)
        for (int t = 0; t < NT; ++t) {
          const int s = t / P, c = t % P;
          if (done[s]) continue;
          const int i0 = lo_i(c), i1 = lo_i(c + 1);
          const T* b = bs[s]; T* x = xs[s];
          for (int ii = 0; ii < i1 - i0; ++ii) {
            const int i = sw == 0 ? i0 + ii : i1 - 1 - ii;
            const int k0 = i * nz;
            const T* xu = (i < nB - 1) ? (i + 1 >= i1 ? &bnd[((size_t)(t + 1) * 2 + 0) * nz] : x + k0 + nz) : x + k0;
            const T* xd = (i > 0) ? (i - 1 < i0 ? &bnd[((size_t)(t - 1) * 2 + 1) * nz] : x + k0 - nz) : x + k0;
            T yp = 0;
            const T* mm = &m[s][k0]; const T* iv = &inv[s][k0]; const T* cpp = &cp[s][k0];
            for (int j = 0; j < nz; ++j) {
              const int k = k0 + j;
              const T r = b[k] + cH[k] * xu[j] + cL[k] * xd[j];
              yp = iv[j] * r + mm[j] * yp; yb[j] = yp;
            }
            T xn = 0;
            for (int j = nz - 1; j >= 0; --j) { xn = yb[j] - cpp[j] * xn; x[k0 + j] = xn; }
          }
        }
      }
#pragma omp for schedule(static)
      for (int t = 0; t < NT; ++t) {
        const int s = t / P, c = t % P;
        if (done[s]) continue;
        const T* b = bs[s]; const T* x = xs[s]; T rr = 0;
        for (int i = lo_i(c); i < lo_i(c + 1); ++i) {
          const int k0 = i * nz;
          const T* xu = (i < nB - 1) ? x + k0 + nz : x + k0;
          const T* xd = (i > 0) ? x + k0 - nz : x + k0;
          for (int j = 0; j < nz; ++j) {
            const int k = k0 + j;
            T r = D[s][k] * x[k] - b[k] - cH[k] * xu[j] - cL[k] * xd[j];
            if (j < nz - 1) r -= g.zup[j] * x[k + 1];
            if (j > 0) r -= g.zdn[j] * x[k - 1];
            rr = std::max(rr, (T)fabs(r));
          }
        }
        resc[t] = rr;
      }
#pragma omp single
      for (int s = 0; s < 2; ++s) {
        if (done[s]) continue;
        ++np[s];
        T r = 0; for (int c = 0; c < P; ++c) r = std::max(r, resc[s * P + c]);
        const T prev = res[s]; res[s] = r;
        if (r < tol[s]) { done[s] = true; flag[s] = 0; }
        else if (sizeof(T) == 4 && r > T(0.9) * prev) { done[s] = true; flag[s] = 1000; }
      }
      if (done[0] && done[1]) break;
    }
  }
  for (int s = 0; s < 2; ++s) n[s] = done[s] ? np[s] + flag[s] : -np[s];
}

template <typename T>
static void policies(const Setup& s, const G<T>& g, const Par<T>& p, Sol<T>& sol, std::vector<T>& tau, std::vector<uint32_t>& mask,
                     bool first, bool count, const Opt& o) {
  const int nB = s.nB, nz = s.nz, N = s.N;
  const uint32_t wmask = (o.freeze_window >= 32) ? 0xFFFFFFFFu : ((1u << o.freeze_window) - 1u);
#pragma omp parallel for schedule(dynamic, 32)
  for (int k = 0; k < N; ++k) {
    const int i = k / nz, j = k % nz;
    const int wp = first ? 9 : sol.which[k];
    const T tk = first ? T(0) : tau[k];
    T S_, c_, d_, u_;
    int w = policy_node<T>(i, j, nB, nz, sol.W.data(), sol.q.data(), g.B.data(), g.y.data(), g.Fz.data(), g.hf.data(),
                           g.hb.data(), p, wp, tk, S_, c_, d_, u_);
    sol.S[k] = S_; sol.c[k] = c_; sol.d[k] = d_; sol.u[k] = u_;
    if (count && o.freeze_flips) {
      const uint32_t ch = (w != sol.which[k]) ? 1u : 0u;
      mask[k] = ((mask[k] << 1) | ch) & wmask;
      if (!sol.frozen[k] && __builtin_popcount(mask[k]) >= o.freeze_flips) { sol.frozen[k] = 1; tau[k] = T(1e30); }
    }
    sol.which[k] = (int8_t)w;
  }
}

template <typename T>
static void solve(const Setup& s, const Opt& o, const std::vector<double>& V0, const std::vector<double>& q0, Sol<T>& sol) {
  const int nB = s.nB, nz = s.nz, N = s.N;
  G<T> g(s);
  Par<T> p = castp<T>(s.p);
  double vr = 0; for (double v : V0) vr += v; vr /= N;
  sol.Vref = vr;
  const T Vref = (T)vr;
  sol.W.resize(N); sol.q.resize(N); sol.S.assign(N, 0); sol.c.assign(N, 0); sol.d.assign(N, 0); sol.u.assign(N, 0);
  for (int k = 0; k < N; ++k) { sol.W[k] = (T)(V0[k] - vr); sol.q[k] = (T)q0[k]; }
  sol.which.assign(N, 9); sol.frozen.assign(N, 0);
  std::vector<T> tau(N, (T)o.sticky); std::vector<uint32_t> mask(N, 0);
  std::vector<T> bV(N), bq(N), ccV(N), ccq(N), cH(N), cL(N), Wp(N), qp(N);
  std::vector<T> D1(N), m1(N), i1(N), c1(N), D2(N), m2(N), i2(N), c2(N);
  const int crows = o.chunk_rows >= 0 ? o.chunk_rows : (nB <= 250 ? 32 : 64);
  const int Pc = crows > 0 ? std::max(1, nB / crows) : 1;
  std::vector<T> Dd[2] = {std::vector<T>(N), std::vector<T>(N)}, md[2] = {std::vector<T>(N), std::vector<T>(N)},
                 invd[2] = {std::vector<T>(N), std::vector<T>(N)}, cpd[2] = {std::vector<T>(N), std::vector<T>(N)}, bnd;
  const T coupon = p.delta + p.lam;
  const T stol = (T)(sizeof(T) == 4 ? std::max(o.sweep_tol, 1e-6) : o.sweep_tol);
  double dt = o.dt0;
  double rV = 0, rq = 0;
  int it;
  bool conv = false;
  double best_m = 1e300, best_q = 1e300; int best_it = 0;
  std::vector<T> Sp, cp_, dp, up_; bool relax_on = false;
  const int probe = getenv("KM_PROBE") ? atoi(getenv("KM_PROBE")) : -1;
  for (it = 1; it <= o.maxit; ++it) {
    const bool first = (it == 1);
    const bool count = !first && dt >= o.dt_max;
    double tp0 = now();
    policies<T>(s, g, p, sol, tau, mask, first, count, o);
    sol.t_policy += now() - tp0;
    T mV = 0, mq = 0;
#pragma omp parallel for reduction(max : mV, mq)
    for (int k = 0; k < N; ++k) {
      T a, b;
      resid_node<T>(k / nz, k % nz, nB, nz, sol.W.data(), sol.q.data(), sol.S.data(), sol.u.data(), sol.d.data(),
                    g.hf.data(), g.hb.data(), g.zup.data(), g.zdn.data(), p, Vref, a, b);
      mV = std::max(mV, a); mq = std::max(mq, b);
    }
    rV = mV; rq = mq;
    if (o.verbose > 1 && (it % 100 == 0 || it == 1)) printf("step %5d dt %.4f resV %.3e resq %.3e\n", it, dt, rV, rq);
    if (rV < o.tolV && rq < o.tolq) { conv = true; break; }
    if (probe >= 0) { T a, b; resid_node<T>(probe / nz, probe % nz, nB, nz, sol.W.data(), sol.q.data(), sol.S.data(), sol.u.data(), sol.d.data(), g.hf.data(), g.hb.data(), g.zup.data(), g.zdn.data(), p, Vref, a, b);
      fprintf(stderr, "%d %.10e %.10e %.10e %.10e %.3e %.3e %d\n", it, (double)sol.q[probe], (double)sol.S[probe], (double)sol.d[probe], (double)sol.c[probe], (double)b, rq, (int)sol.which[probe]); }
    // single precision: tolerances scaled to the precision (relative to |V| and to the unit price), and a stop
    // once the residuals stop improving (they sit at the rounding floor of the scheme)
    if (sizeof(T) == 4 && rV < o.tol32 * 1.1920929e-7 * std::fabs(sol.Vref) && rq < o.tol32 * 1.1920929e-7) { conv = true; break; }
    if (sizeof(T) == 4 && dt >= o.dt_max) {
      // a new low of either residual restarts the window
      if (rV < 0.98 * best_m) { best_m = rV; best_it = it; }
      if (rq < 0.98 * best_q) { best_q = rq; best_it = it; }
      if (it - best_it >= o.stall_window) { sol.stalled = true; break; }
    }
    // policy relaxation (dense grids): the step uses a weighted average of the best response and the previous step's
    // policy; the residuals above are those of the best response, so the fixed point and the stopping rule are unchanged
    if (o.relax < 1.0 && dt >= o.dt_max) {
      if (!relax_on) { Sp = sol.S; cp_ = sol.c; dp = sol.d; up_ = sol.u; relax_on = true; }
      const T w = (T)o.relax;
#pragma omp parallel for
      for (int k = 0; k < N; ++k) {
        sol.S[k] = Sp[k] + w * (sol.S[k] - Sp[k]); sol.c[k] = cp_[k] + w * (sol.c[k] - cp_[k]);
        sol.d[k] = dp[k] + w * (sol.d[k] - dp[k]); sol.u[k] = up_[k] + w * (sol.u[k] - up_[k]);
        Sp[k] = sol.S[k]; cp_[k] = sol.c[k]; dp[k] = sol.d[k]; up_[k] = sol.u[k];
      }
    }
    dt = std::min(dt * o.dt_grow, o.dt_max);
    if (o.accel_on > 0 && rV < o.accel_on && rq < o.accel_on) { static double dta = 0; dta = std::max(dta, dt); dta = std::min(dta * o.accel_grow, o.accel_dtmax); dt = dta; }
    const T idt = (T)(1.0 / dt);
#pragma omp parallel for
    for (int k = 0; k < N; ++k) {
      bV[k] = sol.u[k] - p.rho * Vref + sol.W[k] * idt; ccV[k] = p.rho + idt;
      bq[k] = (T(1) - sol.d[k]) * coupon + sol.q[k] * idt; ccq[k] = p.r + p.delta - p.kappa * coupon * sol.d[k] + idt;
      const int i = k / nz; const T sv = sol.S[k];
      cH[k] = (sv > 0 && i < nB - 1) ? sv / g.hf[i] : T(0); cL[k] = (sv < 0 && i > 0) ? -sv / g.hb[i] : T(0);
    }
    // starting guess of the sweeps: linear extrapolation of the last two iterates (the solve is exact either way)
    if (o.extrap && it > 1) {
#pragma omp parallel for
      for (int k = 0; k < N; ++k) {
        const T w = sol.W[k], qq = sol.q[k];
        sol.W[k] = w + (w - Wp[k]); sol.q[k] = qq + (qq - qp[k]); Wp[k] = w; qp[k] = qq;
      }
    } else { Wp = sol.W; qp = sol.q; }
    int n1 = 0, n2 = 0; T r1, r2;
    (void)r1; (void)r2;
    double ts0 = now();
    if (Pc > 1) {
      int nn[2];
      sweepc<T>(nB, nz, Pc, cH.data(), cL.data(), g, ccV.data(), bV.data(), sol.W.data(), ccq.data(), bq.data(), sol.q.data(), stol,
                o.maxpass, nn, Dd, md, invd, cpd, bnd);
      n1 = nn[0]; n2 = nn[1];
    } else {
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
      n1 = sweep<T>(nB, nz, cH.data(), cL.data(), g, ccV.data(), bV.data(), sol.W.data(), stol, o.maxpass, r1, D1, m1, i1, c1);
#pragma omp section
      n2 = sweep<T>(nB, nz, cH.data(), cL.data(), g, ccq.data(), bq.data(), sol.q.data(), stol, o.maxpass, r2, D2, m2, i2, c2);
    }
    }
    sol.t_sweep += now() - ts0;
    sol.passes += std::abs(n1) % 1000 + std::abs(n2) % 1000;
    sol.unconverged_solves += (n1 < 0) + (n2 < 0); sol.stalled_solves += (n1 > 1000) + (n2 > 1000);
    for (int k = 0; k < N; ++k) sol.q[k] = std::min(T(1), std::max(T(1e-12), sol.q[k]));
  }
  // final policy at the converged (V, q), holding branches as the iteration does
  policies<T>(s, g, p, sol, tau, mask, false, false, o);
  sol.iters = std::min(it, o.maxit); sol.resV = rV; sol.resq = rq; sol.converged = conv;
  sol.n_frozen = 0; for (auto f : sol.frozen) sol.n_frozen += f;
}

template <typename T>
static void simulate(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& P, int& Y, int& L) {
  G<T> g(s);
  std::vector<T> cdf(s.N); double acc = 0, tot = 0;
  for (auto v : sol.g) tot += v;
  for (int k = 0; k < s.N; ++k) { acc += sol.g[k] / tot; cdf[k] = (T)acc; }
  SimArgs<T> a;
  a.S = sol.S.data(); a.d = sol.d.data(); a.c = sol.c.data(); a.q = sol.q.data(); a.Bg = g.B.data(); a.zg = g.z.data(); a.cdf = cdf.data();
  a.nB = s.nB; a.nz = s.nz; a.N = s.N; a.mu = (T)s.mu; a.sigma = (T)s.sigma; a.delta_lam = (T)(s.delta + s.lam); a.r = (T)s.r; a.delta = (T)s.delta;
  a.g0 = (T)s.p.g0; a.g1 = (T)s.p.g1; a.g2 = (T)s.p.g2; a.zt = (T)s.p.zt; a.Fmin = (T)s.p.Fmin;
  a.P = o.n_paths; a.spy = o.spy; a.seed = o.seed; a.Y = o.years;
  if (o.sim_layout == 0) { a.nseg = 1; a.L = o.years; a.burn = o.burn_paper; a.paper_start = 1; a.B0 = (T)(0.25 * (s.B[0] + s.B[s.nB - 1])); a.z0 = 0; }
  else { a.nseg = o.years / o.seg_years; a.L = o.seg_years; a.burn = o.burn_seg; a.paper_start = 0; a.B0 = 0; a.z0 = 0; a.Y = a.nseg * a.L; }
  std::vector<T> buf((size_t)8 * a.P * a.Y);
  a.out = buf.data();
  const int nthr = a.P * a.nseg;
#pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < nthr; ++t) sim_segment<T>(t, a);
  out.assign(buf.begin(), buf.end());
  P = a.P; Y = a.Y; L = a.L;
}

template <typename T>
static void irf(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& nrec);

}  // namespace cpu
