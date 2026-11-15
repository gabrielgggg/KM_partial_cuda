// post.h -- ergodic distribution (banded LU, inverse iteration), ergodic and path moments, impulse responses
// (set-up shared by the back ends; CPU and GPU time stepping), and output.
#pragma once
#include <map>

// ---------------------------------------------------------------- ergodic distribution: A^T g = 0
// Inverse iteration (sigma I - A^T) g' = g with a tiny shift sigma, by a banded LU without pivoting
// (sigma I - A^T is an M-matrix). Band = nz on each side in the ordering k = i*nz + j.
template <typename T> static void ergodic(const Setup& s, Sol<T>& sol) {
  const int nB = s.nB, nz = s.nz, N = s.N, w = nz, bw = 2 * w + 1;
  std::vector<T> hf = castv<T>(s.hf), hb = castv<T>(s.hb), zup = castv<T>(s.zup), zdn = castv<T>(s.zdn);
  const T sig = sizeof(T) == 4 ? T(1e-5) : T(1e-10);
  std::vector<T> a((size_t)N * bw, T(0));
  auto A = [&](int r, int c) -> T& { return a[(size_t)r * bw + (c - r + w)]; };
  for (int i = 0; i < nB; ++i)
    for (int j = 0; j < nz; ++j) {
      const int k = i * nz + j;
      T cLo, cHi, dA;
      at_coefs<T>(i, j, nB, nz, sol.S.data(), hf.data(), hb.data(), zup.data(), zdn.data(), cLo, cHi, dA);
      A(k, k) = sig - dA;
      if (i > 0) A(k, k - nz) = -cLo;
      if (i < nB - 1) A(k, k + nz) = -cHi;
      if (j > 0) A(k, k - 1) = -zup[j - 1];
      if (j < nz - 1) A(k, k + 1) = -zdn[j + 1];
    }
  for (int k = 0; k < N; ++k) {
    const T piv = A(k, k);
    const int rmax = std::min(N - 1, k + w);
    T* rowk = &A(k, k);
    for (int r = k + 1; r <= rmax; ++r) {
      T& ark = A(r, k);
      if (ark == T(0)) continue;
      const T l = ark / piv; ark = l;
      T* rowr = &A(r, k);
      const int n = rmax - k;  // columns k+1 .. k+w (within the band of row k)
      for (int c = 1; c <= n; ++c) rowr[c] -= l * rowk[c];
    }
  }
  std::vector<T> g(N, T(1) / N);
  const int nit = sizeof(T) == 4 ? 4 : 3;
  for (int it = 0; it < nit; ++it) {
    for (int k = 0; k < N; ++k) {  // L y = g
      T v = g[k];
      for (int c = std::max(0, k - w); c < k; ++c) v -= A(k, c) * g[c];
      g[k] = v;
    }
    for (int k = N - 1; k >= 0; --k) {  // U x = y
      T v = g[k];
      for (int c = k + 1; c <= std::min(N - 1, k + w); ++c) v -= A(k, c) * g[c];
      g[k] = v / A(k, k);
    }
    double sm = 0; for (int k = 0; k < N; ++k) { g[k] = std::max(T(0), g[k]); sm += g[k]; }
    for (int k = 0; k < N; ++k) g[k] = (T)(g[k] / sm);
  }
  double res = 0;
  for (int i = 0; i < nB; ++i)
    for (int j = 0; j < nz; ++j)
      res = std::max(res, (double)fabs(at_apply<T>(i, j, nB, nz, sol.S.data(), hf.data(), hb.data(), zup.data(), zdn.data(), g.data())));
  sol.g = g; sol.ergodic_res = res;
}

// ---------------------------------------------------------------- moments
struct Moments { std::map<std::string, double> E, Sm, Ssd, M; double loss = 0; };
static const char* TARGETS[] = {"pd_freq", "pd_mean_cond", "pd_sd_cond", "debt_y_mean", "debt_y_sd", "ds_y_mean", "ds_y_sd", "dd_y_mean", "spread_sd", "y_persist", "y_sd"};
static const double TDATA[] = {0.36, 0.38, 0.22, 0.32, 0.18, 0.036, 0.021, 0.049, 0.041, 0.89, 0.10};
static const double TW[] = {1, 1, 1, 1, 1, 1, 1, 3, 1.25, 1, 1};
static const char* EKEYS[] = {"debt_y_mean", "debt_y_sd", "ds_y_mean", "ds_y_sd", "dd_y_mean", "spread_sd", "spread_mean", "corr_spread_y", "corr_spread_debt"};
static const char* SKEYS[] = {"pd_freq", "pd_mean_cond", "pd_sd_cond", "y_persist", "y_sd", "dc_y_mean_cond", "dc_y_sd_cond", "c_sd_rel"};

template <typename T> static std::vector<double> net_output(const Setup& s, const Sol<T>& sol) {
  std::vector<double> y(s.N);
  for (int i = 0; i < s.nB; ++i)
    for (int j = 0; j < s.nz; ++j) {
      const int k = i * s.nz + j; const double d = sol.d[k];
      y[k] = (d <= 0 ? 1.0 : (1 - s.p.g0 * pow(d, s.p.g1)) * s.Fz[j]) * s.y[j];
    }
  return y;
}

template <typename T> static Moments ergodic_moments(const Setup& s, const Sol<T>& sol) {
  Moments R; auto& E = R.E;
  const int N = s.N;
  std::vector<double> w(N); double tot = 0;
  for (int k = 0; k < N; ++k) tot += sol.g[k];
  for (int k = 0; k < N; ++k) w[k] = sol.g[k] / tot;
  std::vector<double> y = net_output(s, sol);
  const double coupon = s.delta + s.lam;
  std::vector<double> debt_y(N), dd_y(N), ds_y(N), dc_y(N), sp(N), ly(N), lc(N);
  for (int k = 0; k < N; ++k) {
    const double B = s.B[k / s.nz], d = sol.d[k];
    debt_y[k] = B / y[k]; dd_y[k] = coupon * B / y[k]; ds_y[k] = (1 - d) * dd_y[k]; dc_y[k] = d * dd_y[k];
    sp[k] = coupon / (double)sol.q[k] - (s.delta + s.r); ly[k] = log(y[k]); lc[k] = log((double)sol.c[k]);
  }
  auto mean = [&](const std::vector<double>& x) { double m = 0; for (int k = 0; k < N; ++k) m += w[k] * x[k]; return m; };
  auto sd = [&](const std::vector<double>& x) { double m = mean(x), v = 0; for (int k = 0; k < N; ++k) v += w[k] * (x[k] - m) * (x[k] - m); return sqrt(v); };
  auto cov = [&](const std::vector<double>& x, const std::vector<double>& z) { double mx = mean(x), mz = mean(z), v = 0; for (int k = 0; k < N; ++k) v += w[k] * (x[k] - mx) * (z[k] - mz); return v; };
  E["debt_y_mean"] = mean(debt_y); E["debt_y_sd"] = sd(debt_y); E["ds_y_mean"] = mean(ds_y); E["ds_y_sd"] = sd(ds_y);
  E["dd_y_mean"] = mean(dd_y); E["dd_y_sd"] = sd(dd_y); E["spread_mean"] = mean(sp); E["spread_sd"] = sd(sp);
  double wpd = 0, wd = 0, wdc = 0, mfd = 0, mtop = 0, vtop = 0;
  const double Btop = s.B[s.nB - 1] - 0.1 * (s.B[s.nB - 1] - s.B[0]), msp = E["spread_mean"];
  for (int k = 0; k < N; ++k) {
    if (sol.d[k] > 1e-10) { wpd += w[k]; wd += w[k] * sol.d[k]; wdc += w[k] * dc_y[k]; }
    if (sol.d[k] > 0.999) mfd += w[k];
    if (s.B[k / s.nz] > Btop) { mtop += w[k]; vtop += w[k] * (sp[k] - msp) * (sp[k] - msp); }
  }
  E["pd_point_freq"] = wpd; E["d_mean_cond"] = wd / std::max(wpd, 1e-300); E["dc_y_mean_cond"] = wdc / std::max(wpd, 1e-300);
  E["c_sd_rel"] = sd(lc) / sd(ly);
  E["corr_spread_y"] = cov(sp, ly) / (E["spread_sd"] * sd(ly) + 1e-300);
  E["corr_spread_debt"] = cov(sp, debt_y) / (E["spread_sd"] * E["debt_y_sd"] + 1e-300);
  E["mass_top"] = mtop; E["spread_var_share_top"] = vtop / (E["spread_sd"] * E["spread_sd"] + 1e-300); E["mass_full_default"] = mfd;
  double EB = 0; for (int k = 0; k < N; ++k) EB += w[k] * s.B[k / s.nz];
  E["EB"] = EB; E["atom0"] = 0; for (int j = 0; j < s.nz; ++j) E["atom0"] += w[j];
  return R;
}

// moments of one path with the paper's definitions; lag-one pairs that straddle two segments are skipped
static std::map<std::string, double> path_moments_one(const double* f, long PY, long off, int Y, int L) {
  const double *y = f + 0 * PY + off, *c = f + 1 * PY + off, *dd = f + 2 * PY + off, *ds = f + 3 * PY + off, *dc = f + 4 * PY + off,
               *pd = f + 5 * PY + off, *B = f + 6 * PY + off, *sp = f + 7 * PY + off;
  std::vector<double> ly(Y), lc(Y);
  for (int t = 0; t < Y; ++t) { ly[t] = log(y[t]); lc[t] = log(c[t]); }
  auto detrend = [&](std::vector<double>& x) {
    double st = 0, sx = 0, stt = 0, stx = 0;
    for (int t = 0; t < Y; ++t) { st += t; sx += x[t]; stt += (double)t * t; stx += t * x[t]; }
    const double bb = (Y * stx - st * sx) / (Y * stt - st * st), aa = (sx - bb * st) / Y;
    for (int t = 0; t < Y; ++t) x[t] -= aa + bb * t;
  };
  detrend(ly); detrend(lc);
  auto msd = [&](auto get, auto sel, double& m, double& sd, double& n) {
    double s1 = 0, s2 = 0; n = 0;
    for (int t = 0; t < Y; ++t) if (sel(t)) { double v = get(t); s1 += v; n += 1; }
    m = n > 0 ? s1 / n : 0;
    for (int t = 0; t < Y; ++t) if (sel(t)) { double v = get(t) - m; s2 += v * v; }
    sd = n > 0 ? sqrt(s2 / n) : 0;
  };
  auto all = [](int) { return true; };
  auto inpd = [&](int t) { return pd[t] > 0.0; };
  std::map<std::string, double> M; double m, s, n;
  msd([&](int t) { return 1.0; }, inpd, m, s, n); M["pd_freq"] = n / Y;
  msd([&](int t) { return pd[t]; }, inpd, m, s, n); M["pd_mean_cond"] = m; M["pd_sd_cond"] = s;
  msd([&](int t) { return B[t] / y[t]; }, all, m, s, n); M["debt_y_mean"] = m; M["debt_y_sd"] = s;
  msd([&](int t) { return ds[t] / y[t]; }, all, m, s, n); M["ds_y_mean"] = m; M["ds_y_sd"] = s;
  msd([&](int t) { return dd[t] / y[t]; }, all, m, s, n); M["dd_y_mean"] = m;
  msd([&](int t) { return sp[t]; }, all, m, s, n); M["spread_sd"] = s; M["spread_mean"] = m;
  double ysd; msd([&](int t) { return ly[t]; }, all, m, ysd, n); M["y_sd"] = ysd;
  {  // lag-one correlation over pairs inside segments
    double a1 = 0, a2 = 0, b1 = 0, b2 = 0, ab = 0, np = 0;
    for (int t = 1; t < Y; ++t) { if (t % L == 0) continue; double u = ly[t], v = ly[t - 1]; a1 += u; a2 += u * u; b1 += v; b2 += v * v; ab += u * v; np += 1; }
    double cu = a2 / np - (a1 / np) * (a1 / np), cv = b2 / np - (b1 / np) * (b1 / np), cuv = ab / np - (a1 / np) * (b1 / np);
    M["y_persist"] = cuv / sqrt(cu * cv);
  }
  msd([&](int t) { return dc[t] / y[t]; }, inpd, m, s, n); M["dc_y_mean_cond"] = m; M["dc_y_sd_cond"] = s;
  auto corr = [&](auto fx, auto fy) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int t = 0; t < Y; ++t) { double u = fx(t), v = fy(t); sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; }
    double mx = sx / Y, my = sy / Y;
    return (sxy / Y - mx * my) / sqrt((sxx / Y - mx * mx) * (syy / Y - my * my));
  };
  M["corr_spread_y"] = corr([&](int t) { return sp[t]; }, [&](int t) { return ly[t]; });
  M["corr_spread_debt"] = corr([&](int t) { return sp[t]; }, [&](int t) { return B[t] / y[t]; });
  double lcsd; msd([&](int t) { return lc[t]; }, all, m, lcsd, n); M["c_sd_rel"] = lcsd / ysd;
  return M;
}

static void path_moments(const std::vector<double>& f, int P, int Y, int L, Moments& R) {
  std::vector<std::map<std::string, double>> Ms(P);
  const long PY = (long)P * Y;
#pragma omp parallel for
  for (int p = 0; p < P; ++p) Ms[p] = path_moments_one(f.data(), PY, (long)p * Y, Y, L);
  for (auto& kv : Ms[0]) {
    double m = 0, v = 0;
    for (int p = 0; p < P; ++p) m += Ms[p][kv.first];
    m /= P;
    for (int p = 0; p < P; ++p) v += (Ms[p][kv.first] - m) * (Ms[p][kv.first] - m);
    R.Sm[kv.first] = m; R.Ssd[kv.first] = sqrt(v / P);
  }
  for (auto k : EKEYS) R.M[k] = R.E[k];
  for (auto k : SKEYS) R.M[k] = R.Sm[k];
  R.loss = 0;
  for (int i = 0; i < 11; ++i) R.loss += TW[i] * pow((R.M[TARGETS[i]] - TDATA[i]) / TDATA[i], 2);
}

// ---------------------------------------------------------------- impulse responses: set-up
// runs: 0 ergodic with the endowment shifted by the shock; 1..6 (state m: shock, control) from point masses at
// (m E[B], 0) for m = 1, 0.75, 1.25; the control of run 0 is the ergodic expectation (constant).
template <typename T> struct IrfSetup {
  int R = 7, NF = 8, nT = 0; double h = 0;
  std::vector<double> g0, F, ctrl;   // g0: R*N; F: NF*N (y, yn, c, d, B, spread, issue, defc); ctrl: NF
};
static void shift_in_z(const Setup& s, const double* g, double* out, double shift) {
  const int nB = s.nB, nz = s.nz; const double hz = s.z[1] - s.z[0];
  for (int k = 0; k < s.N; ++k) out[k] = 0;
  for (int jk = 0; jk < nz; ++jk) {
    double pos = (s.z[jk] + shift - s.z[0]) / hz;
    for (int i = 0; i < nB; ++i) {
      double v = g[i * nz + jk];
      if (pos <= 0) out[i * nz] += v;
      else if (pos >= nz - 1) out[i * nz + nz - 1] += v;
      else { int j = (int)floor(pos); double ww = pos - j; out[i * nz + j] += (1 - ww) * v; out[i * nz + j + 1] += ww * v; }
    }
  }
}
template <typename T> static IrfSetup<T> irf_setup(const Setup& s, const Opt& o, const Sol<T>& sol) {
  IrfSetup<T> I; const int N = s.N, nz = s.nz;
  I.nT = (int)lround(o.irf_T / o.irf_dt); I.h = 0.5 * o.irf_dt;
  std::vector<double> ge(N); double tot = 0;
  for (int k = 0; k < N; ++k) tot += sol.g[k];
  for (int k = 0; k < N; ++k) ge[k] = sol.g[k] / tot;
  std::vector<double> yn = net_output(s, sol);
  const double coupon = s.delta + s.lam;
  I.F.assign((size_t)I.NF * N, 0);
  for (int k = 0; k < N; ++k) {
    const double B = s.B[k / nz], d = sol.d[k];
    I.F[0 * N + k] = s.y[k % nz]; I.F[1 * N + k] = yn[k]; I.F[2 * N + k] = sol.c[k]; I.F[3 * N + k] = d; I.F[4 * N + k] = B;
    I.F[5 * N + k] = coupon / (double)sol.q[k] - (s.delta + s.r);
    I.F[6 * N + k] = sol.S[k] + s.delta * B - s.kappa * coupon * d * B; I.F[7 * N + k] = d * coupon * B;
  }
  I.ctrl.assign(I.NF, 0);
  for (int f = 0; f < I.NF; ++f) for (int k = 0; k < N; ++k) I.ctrl[f] += ge[k] * I.F[(size_t)f * N + k];
  double EB = 0; for (int k = 0; k < N; ++k) EB += ge[k] * s.B[k / nz];
  I.g0.assign((size_t)I.R * N, 0);
  shift_in_z(s, ge.data(), &I.g0[0], o.irf_shock);
  const double mult[3] = {1.0, 0.75, 1.25};
  for (int st = 0; st < 3; ++st) {
    std::vector<double> pm(N, 0);
    double B0 = mult[st] * EB, z0 = 0.0;
    int iB = (int)(std::lower_bound(s.B.begin(), s.B.end(), B0) - s.B.begin()) - 1; iB = std::min(std::max(iB, 0), s.nB - 2);
    int jz = (int)(std::lower_bound(s.z.begin(), s.z.end(), z0) - s.z.begin()) - 1; jz = std::min(std::max(jz, 0), nz - 2);
    double wB = std::min(1.0, std::max(0.0, (B0 - s.B[iB]) / (s.B[iB + 1] - s.B[iB])));
    double wz = std::min(1.0, std::max(0.0, (z0 - s.z[jz]) / (s.z[jz + 1] - s.z[jz])));
    pm[iB * nz + jz] += (1 - wB) * (1 - wz); pm[(iB + 1) * nz + jz] += wB * (1 - wz);
    pm[iB * nz + jz + 1] += (1 - wB) * wz; pm[(iB + 1) * nz + jz + 1] += wB * wz;
    shift_in_z(s, pm.data(), &I.g0[(size_t)(1 + 2 * st) * N], o.irf_shock);
    std::copy(pm.begin(), pm.end(), I.g0.begin() + (size_t)(2 + 2 * st) * N);
  }
  return I;
}
// z-line Thomas factors of (I - h A^T)
template <typename T> static void irf_factors(const Setup& s, const Sol<T>& sol, double h, std::vector<T>& m, std::vector<T>& inv, std::vector<T>& cp) {
  const int nB = s.nB, nz = s.nz, N = s.N;
  std::vector<T> hf = castv<T>(s.hf), hb = castv<T>(s.hb), zup = castv<T>(s.zup), zdn = castv<T>(s.zdn);
  m.resize(N); inv.resize(N); cp.resize(N);
  std::vector<T> D(nz), lo(nz), upc(nz);
  for (int j = 0; j < nz; ++j) { lo[j] = j > 0 ? (T)h * zup[j - 1] : T(0); upc[j] = j < nz - 1 ? (T)h * zdn[j + 1] : T(0); }
  for (int i = 0; i < nB; ++i) {
    for (int j = 0; j < nz; ++j) { T a, b, dA; at_coefs<T>(i, j, nB, nz, sol.S.data(), hf.data(), hb.data(), zup.data(), zdn.data(), a, b, dA); D[j] = T(1) - (T)h * dA; }
    thomas_factor<T>(nz, D.data(), lo.data(), upc.data(), &m[i * nz], &inv[i * nz], &cp[i * nz]);
  }
}
static int irf_iters_default(const Opt& o, int fp) { return o.irf_iters > 0 ? o.irf_iters : 3; }

namespace cpu {
template <typename T>
static void irf(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& nrec) {
  IrfSetup<T> I = irf_setup<T>(s, o, sol);
  const int N = s.N, nB = s.nB, nz = s.nz, R = I.R, NF = I.NF;
  std::vector<T> m, inv, cp; irf_factors<T>(s, sol, I.h, m, inv, cp);
  std::vector<T> hf = castv<T>(s.hf), hb = castv<T>(s.hb), zup = castv<T>(s.zup), zdn = castv<T>(s.zdn);
  std::vector<T> g = castv<T>(I.g0), rhs((size_t)R * N), F = castv<T>(I.F);
  nrec = I.nT + 1;
  out.assign((size_t)(R + 1) * NF * nrec, 0);
  const int iters = irf_iters_default(o, sizeof(T) == 4 ? 32 : 64);
  const T h = (T)I.h;
  std::vector<T> scale(R, T(1));
  // coefficients of the neighbouring debt rows in each row system (as in gpu::irf)
  std::vector<T> cIm(N, T(0)), cIp(N, T(0));
  for (int i = 0; i < nB; ++i) for (int j = 0; j < nz; ++j) {
    const int k = i * nz + j;
    if (i > 0) { T sv = sol.S[k - nz]; if (sv > T(0)) cIm[k] = h * sv / hf[i - 1]; }
    if (i < nB - 1) { T sv = sol.S[k + nz]; if (sv < T(0)) cIp[k] = -h * sv / hb[i + 1]; }
  }
  std::vector<double> acc((size_t)R * 9);
#pragma omp parallel
  {
    std::vector<T> yb((size_t)nz);
    std::vector<double> la((size_t)R * 9);
    for (int t = 0; t <= I.nT; ++t) {
      // clip, total mass and the expectations of every run in one pass over the fields; the normalization is
      // folded into the next right-hand side
#pragma omp single
      std::fill(acc.begin(), acc.end(), 0.0);
      std::fill(la.begin(), la.end(), 0.0);
#pragma omp for schedule(static) nowait
      for (int k = 0; k < N; ++k) {
        T Fk[8];
        for (int f = 0; f < 8; ++f) Fk[f] = F[(size_t)f * N + k];
        for (int r = 0; r < R; ++r) {
          T& gv = g[(size_t)r * N + k];
          const T v = std::max(T(0), gv); gv = v;
          double* a = &la[(size_t)r * 9];
          a[8] += v;
          for (int f = 0; f < 8; ++f) a[f] += (double)(v * Fk[f]);
        }
      }
#pragma omp critical
      for (int i = 0; i < R * 9; ++i) acc[i] += la[i];
#pragma omp barrier
#pragma omp single
      for (int r = 0; r < R; ++r) {
        for (int f = 0; f < NF; ++f) out[((size_t)r * NF + f) * nrec + t] = acc[r * 9 + f] / acc[r * 9 + 8];
        scale[r] = (T)(1.0 / acc[r * 9 + 8]);
      }
      if (t == I.nT) break;
#pragma omp for schedule(static)
      for (int k = 0; k < N; ++k) {
        const int i = k / nz, j = k % nz;
        T cLo, cHi, dA;
        at_coefs<T>(i, j, nB, nz, sol.S.data(), hf.data(), hb.data(), zup.data(), zdn.data(), cLo, cHi, dA);
        const T zu = j > 0 ? zup[j - 1] : T(0), zd = j < nz - 1 ? zdn[j + 1] : T(0);
        for (int r = 0; r < R; ++r) {
          const T* gr = &g[(size_t)r * N];
          T v = dA * gr[k];
          if (i > 0) v += cLo * gr[k - nz];
          if (i < nB - 1) v += cHi * gr[k + nz];
          if (j > 0) v += zu * gr[k - 1];
          if (j < nz - 1) v += zd * gr[k + 1];
          rhs[(size_t)r * N + k] = scale[r] * (gr[k] + h * v);
        }
      }
      for (int it = 0; it < iters; ++it)
        for (int par = 0; par < 2; ++par) {
          const int nrow = (nB - par + 1) / 2;
#pragma omp for schedule(static)
          for (int ii = 0; ii < nrow; ++ii) {
            const int i = 2 * ii + par, k0 = i * nz;
            for (int r = 0; r < R; ++r) {
              const T* rr = &rhs[(size_t)r * N]; T* x = &g[(size_t)r * N];
              T yp = 0;
              for (int j = 0; j < nz; ++j) {
                const int k = k0 + j;
                T a = rr[k];
                if (i > 0) a += cIm[k] * x[k - nz];
                if (i < nB - 1) a += cIp[k] * x[k + nz];
                yp = inv[k] * a + m[k] * yp; yb[j] = yp;
              }
              T xn = 0;
              for (int j = nz - 1; j >= 0; --j) { xn = yb[j] - cp[k0 + j] * xn; x[k0 + j] = xn; }
            }
          }
        }
    }
  }
  for (int f = 0; f < NF; ++f) for (int t = 0; t < nrec; ++t) out[((size_t)R * NF + f) * nrec + t] = I.ctrl[f];
}
}  // namespace cpu

namespace gpu {
// clip, total mass and the expectations in one pass; the normalization is applied in the next right-hand side
template <typename T> __global__ void k_irf_post(int N, int NF, int nrec, int t, T* g, const T* F, double* out, T* scale) {
  const int r = blockIdx.x; T* gr = g + (size_t)r * N;
  __shared__ T red[9 * 32];
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  T acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    const T v = fmax(T(0), gr[k]); gr[k] = v; acc[8] += v;
#pragma unroll
    for (int f = 0; f < 8; ++f) acc[f] += v * F[(size_t)f * N + k];
  }
#pragma unroll
  for (int f = 0; f < 9; ++f) { T a = warpsum(acc[f]); if (lane == 0) red[f * nw + wid] = a; }
  __syncthreads();
  if (threadIdx.x < 9) { T a = 0; for (int i = 0; i < nw; ++i) a += red[threadIdx.x * nw + i]; red[8 * 32 + threadIdx.x] = a; }
  __syncthreads();
  const T sm = red[8 * 32 + 8];
  if (threadIdx.x < NF) out[((size_t)r * NF + threadIdx.x) * nrec + t] = (double)(red[8 * 32 + threadIdx.x] / sm);
  if (threadIdx.x == 0) scale[r] = T(1) / sm;
}
template <typename T> __global__ void k_irf_rhs(int R, int nB, int nz, const T* S, DG<T> gd, const T* g, T* rhs, T h, const T* scale) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; const int N = nB * nz;
  if (idx >= (long)R * N) return;
  const int r = idx / N, k = idx % N;
  rhs[idx] = scale[r] * (g[idx] + h * at_apply<T>(k / nz, k % nz, nB, nz, S, gd.hf, gd.hb, gd.zup, gd.zdn, g + (size_t)r * N));
}
// ---- larger grids: one pass over the coefficients serves all the runs
constexpr int RMAX = 8;
// clip, mass and expectations of every run in one pass over the fields; block sums are added into acc (double)
template <typename T> __global__ void k_irf_post_all(int N, int R, const T* __restrict__ F, T* g, double* acc) {
  __shared__ double red[RMAX * 9];
  for (int i = threadIdx.x; i < R * 9; i += blockDim.x) red[i] = 0;
  __syncthreads();
  double a[RMAX][9];
#pragma unroll
  for (int r = 0; r < RMAX; ++r)
#pragma unroll
    for (int f = 0; f < 9; ++f) a[r][f] = 0;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x) {
    T Fk[8];
#pragma unroll
    for (int f = 0; f < 8; ++f) Fk[f] = F[(size_t)f * N + k];
#pragma unroll
    for (int r = 0; r < RMAX; ++r) {
      if (r >= R) break;
      const T v = fmax(T(0), g[(size_t)r * N + k]); g[(size_t)r * N + k] = v;
      a[r][8] += (double)v;
#pragma unroll
      for (int f = 0; f < 8; ++f) a[r][f] += (double)(v * Fk[f]);
    }
  }
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int r = 0; r < RMAX; ++r) {
    if (r >= R) break;
#pragma unroll
    for (int f = 0; f < 9; ++f) {
      double v = a[r][f];
      for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (lane == 0) atomicAdd(&red[r * 9 + f], v);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < R * 9; i += blockDim.x) atomicAdd(&acc[i], red[i]);
}
template <typename T> __global__ void k_irf_fin(int R, int NF, int nrec, int t, double* acc, double* out, T* scale) {
  const int i = threadIdx.x;
  if (i < R * NF) { const int r = i / NF, f = i % NF; out[((size_t)r * NF + f) * nrec + t] = acc[r * 9 + f] / acc[r * 9 + 8]; }
  __syncthreads();
  if (i < R) scale[i] = (T)(1.0 / acc[i * 9 + 8]);
  __syncthreads();
  for (int k = i; k < R * 9; k += blockDim.x) acc[k] = 0;
}
template <typename T> __global__ void k_irf_rhs_all(int R, int nB, int nz, const T* S, DG<T> gd, const T* g, T* rhs, T h, const T* scale) {
  const int N = nB * nz;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  const int i = k / nz, j = k % nz;
  T cLo, cHi, dA;
  at_coefs<T>(i, j, nB, nz, S, gd.hf, gd.hb, gd.zup, gd.zdn, cLo, cHi, dA);
  const T zu = j > 0 ? gd.zup[j - 1] : T(0), zd = j < nz - 1 ? gd.zdn[j + 1] : T(0);
  for (int r = 0; r < R; ++r) {
    const T* gr = g + (size_t)r * N;
    T v = dA * gr[k];
    if (i > 0) v += cLo * gr[k - nz];
    if (i < nB - 1) v += cHi * gr[k + nz];
    if (j > 0) v += zu * gr[k - 1];
    if (j < nz - 1) v += zd * gr[k + 1];
    rhs[(size_t)r * N + k] = scale[r] * (gr[k] + h * v);
  }
}
// warp per debt row, all runs in turn with the row's coefficients held in registers
template <typename T, int K>
__global__ void __launch_bounds__(128) k_irf_zebra_all(int R, int nB, int nz, int par, const T* __restrict__ cIm, const T* __restrict__ cIp,
                                                       const T* __restrict__ m, const T* __restrict__ inv, const T* __restrict__ cp,
                                                       const T* __restrict__ rhs, T* g) {
  const int nrow = (nB - par + 1) / 2;
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= nrow) return;
  const int i = 2 * w + par; const size_t N = (size_t)nB * nz;
  T cm[K], cq[K], mm[K], iv[K], cc[K];
#pragma unroll
  for (int e = 0; e < K; ++e) {
    const int j = lane * K + e;
    if (j < nz) { const int k = i * nz + j; cm[e] = cIm[k]; cq[e] = cIp[k]; mm[e] = m[k]; iv[e] = inv[k]; cc[e] = cp[k]; }
    else { cm[e] = 0; cq[e] = 0; mm[e] = 0; iv[e] = 0; cc[e] = 0; }
  }
  for (int r = 0; r < R; ++r) {
    const T* rr = rhs + r * N; T* x = g + r * N;
    T rv[K];
#pragma unroll
    for (int e = 0; e < K; ++e) {
      const int j = lane * K + e;
      if (j < nz) {
        const int k = i * nz + j;
        T a = rr[k];
        if (i > 0) a += cm[e] * x[k - nz];
        if (i < nB - 1) a += cq[e] * x[k + nz];
        rv[e] = a;
      } else rv[e] = 0;
    }
    T M = 1, E = 0;
#pragma unroll
    for (int e = 0; e < K; ++e) { E = mm[e] * E + iv[e] * rv[e]; M = mm[e] * M; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      T Mo = __shfl_up_sync(0xffffffffu, M, o), Eo = __shfl_up_sync(0xffffffffu, E, o);
      if (lane >= o) { E = M * Eo + E; M = M * Mo; }
    }
    T yv = __shfl_up_sync(0xffffffffu, E, 1);
    if (lane == 0) yv = 0;
    T ys[K];
#pragma unroll
    for (int e = 0; e < K; ++e) { yv = mm[e] * yv + iv[e] * rv[e]; ys[e] = yv; }
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
    for (int e = K - 1; e >= 0; --e) { xv = ys[e] - cc[e] * xv; const int j = lane * K + e; if (j < nz) x[i * nz + j] = xv; }
  }
}
// warp-per-row version of irf_row: lane l owns z-nodes l*K..l*K+K-1, the two Thomas recurrences are affine scans
template <typename T, int K>
__global__ void __launch_bounds__(128) k_irf_zebra_w(int R, int nB, int nz, int par, const T* __restrict__ cIm, const T* __restrict__ cIp,
                                                     const T* __restrict__ m, const T* __restrict__ inv, const T* __restrict__ cp,
                                                     const T* __restrict__ rhs, T* g) {
  const int nrow = (nB - par + 1) / 2;
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= R * nrow) return;
  const int r = w / nrow, i = 2 * (w % nrow) + par; const size_t N = (size_t)nB * nz;
  const T* rr = rhs + r * N; T* x = g + r * N;
  T rv[K], mm[K], iv[K], cc[K];
#pragma unroll
  for (int e = 0; e < K; ++e) {
    const int j = lane * K + e;
    if (j < nz) {
      const int k = i * nz + j;
      T a = rr[k];
      if (i > 0) a += cIm[k] * x[k - nz];
      if (i < nB - 1) a += cIp[k] * x[k + nz];
      rv[e] = a; mm[e] = m[k]; iv[e] = inv[k]; cc[e] = cp[k];
    } else { rv[e] = 0; mm[e] = 0; iv[e] = 0; cc[e] = 0; }
  }
  T M = 1, E = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) { E = mm[e] * E + iv[e] * rv[e]; M = mm[e] * M; }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    T Mo = __shfl_up_sync(0xffffffffu, M, o), Eo = __shfl_up_sync(0xffffffffu, E, o);
    if (lane >= o) { E = M * Eo + E; M = M * Mo; }
  }
  T yv = __shfl_up_sync(0xffffffffu, E, 1);
  if (lane == 0) yv = 0;
  T ys[K];
#pragma unroll
  for (int e = 0; e < K; ++e) { yv = mm[e] * yv + iv[e] * rv[e]; ys[e] = yv; }
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
  for (int e = K - 1; e >= 0; --e) { xv = ys[e] - cc[e] * xv; const int j = lane * K + e; if (j < nz) x[i * nz + j] = xv; }
}
template <typename T>
static void irf(const Setup& s, const Opt& o, const Sol<T>& sol, std::vector<double>& out, int& nrec) {
  IrfSetup<T> I = irf_setup<T>(s, o, sol);
  const int N = s.N, nB = s.nB, nz = s.nz, R = I.R, NF = I.NF;
  std::vector<T> m, inv, cp; irf_factors<T>(s, sol, I.h, m, inv, cp);
  // coefficients of the neighbouring debt rows in each row system
  std::vector<T> cIm(N, T(0)), cIp(N, T(0));
  {
    std::vector<T> hf = castv<T>(s.hf), hb = castv<T>(s.hb);
    const T hh = (T)I.h;
    for (int i = 0; i < nB; ++i) for (int j = 0; j < nz; ++j) {
      const int k = i * nz + j;
      if (i > 0) { T sv = sol.S[k - nz]; if (sv > T(0)) cIm[k] = hh * sv / hf[i - 1]; }
      if (i < nB - 1) { T sv = sol.S[k + nz]; if (sv < T(0)) cIp[k] = -hh * sv / hb[i + 1]; }
    }
  }
  T *dIm = up(cIm), *dIp = up(cIp), *dscale = dalloc<T>(R);
  const int Kz = (nz + 31) / 32;
  DG<T> gd = upgrid<T>(s);
  T *dS = up(sol.S), *dm = up(m), *dinv = up(inv), *dcp = up(cp), *dg = up(castv<T>(I.g0)), *dF = up(castv<T>(I.F)), *drhs = dalloc<T>((size_t)R * N);
  nrec = I.nT + 1;
  double* dout = dalloc<double>((size_t)R * NF * nrec);
  const int iters = irf_iters_default(o, sizeof(T) == 4 ? 32 : 64);
  const T h = (T)I.h;
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t graph; cudaGraphExec_t gexec;
  // capture one time step (rhs, zebra relaxation) as a graph; the recording kernel carries the date
  const bool all = nB >= o.irf_all_rows;   // larger grids: one pass over the coefficients for all runs
  double* dacc = dalloc<double>((size_t)RMAX * 9); CK(cudaMemset(dacc, 0, sizeof(double) * RMAX * 9));
  if (all) {
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k_irf_rhs_all<T><<<(N + 127) / 128, 128, 0, st>>>(R, nB, nz, dS, gd, dg, drhs, h, dscale);
    for (int it = 0; it < iters; ++it)
      for (int par = 0; par < 2; ++par) {
        const int nrow = (nB - par + 1) / 2, nbk = (nrow * 32 + 127) / 128;
        switch (Kz) {
          case 1: k_irf_zebra_all<T, 1><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
          case 2: k_irf_zebra_all<T, 2><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
          case 3: k_irf_zebra_all<T, 3><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
          default: k_irf_zebra_all<T, 4><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
        }
      }
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&gexec, graph, 0));
    int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    const int pblk = std::min((N + 127) / 128, nsm * 8);
    for (int t = 0; t <= I.nT; ++t) {
      k_irf_post_all<T><<<pblk, 128, 0, st>>>(N, R, dF, dg, dacc);
      k_irf_fin<T><<<1, 64, 0, st>>>(R, NF, nrec, t, dacc, dout, dscale);
      if (t == I.nT) break;
      CK(cudaGraphLaunch(gexec, st));
    }
  } else {
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  k_irf_rhs<T><<<(int)(((long)R * N + 255) / 256), 256, 0, st>>>(R, nB, nz, dS, gd, dg, drhs, h, dscale);
  for (int it = 0; it < iters; ++it)
    for (int par = 0; par < 2; ++par) {
      const int nrow = (nB - par + 1) / 2;
      const int nbk = (R * nrow * 32 + 127) / 128;
      switch (Kz) {
        case 1: k_irf_zebra_w<T, 1><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
        case 2: k_irf_zebra_w<T, 2><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
        case 3: k_irf_zebra_w<T, 3><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
        default: k_irf_zebra_w<T, 4><<<nbk, 128, 0, st>>>(R, nB, nz, par, dIm, dIp, dm, dinv, dcp, drhs, dg); break;
      }
    }
  CK(cudaStreamEndCapture(st, &graph));
  CK(cudaGraphInstantiate(&gexec, graph, 0));
  for (int t = 0; t <= I.nT; ++t) {
    k_irf_post<T><<<R, 256, 0, st>>>(N, NF, nrec, t, dg, dF, dout, dscale);
    if (t == I.nT) break;
    CK(cudaGraphLaunch(gexec, st));
  }
  }
  CK(cudaStreamSynchronize(st));
  cudaFree(dacc);
  std::vector<double> o1; down(o1, dout, (size_t)R * NF * nrec);
  out.assign((size_t)(R + 1) * NF * nrec, 0);
  std::copy(o1.begin(), o1.end(), out.begin());
  for (int f = 0; f < NF; ++f) for (int t = 0; t < nrec; ++t) out[((size_t)R * NF + f) * nrec + t] = I.ctrl[f];
  for (T* ptr : {dS, dm, dinv, dcp, dg, dF, drhs, dIm, dIp, dscale}) cudaFree(ptr);
  cudaFree(dout); cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph); cudaStreamDestroy(st);
}
}  // namespace gpu

// ---------------------------------------------------------------- output
template <typename T>
static void write_all(const Setup& s, const Opt& o, const Sol<T>& sol, const Moments& M, const Timer& tm, const std::vector<double>& irf, int nrec) {
  const std::string d = o.out + "/";
  const long nB = s.nB, nz = s.nz;
  std::vector<double> V(s.N);
  for (int k = 0; k < s.N; ++k) V[k] = (double)sol.W[k] + sol.Vref;
  write_npy(d + "B.npy", s.B, {nB}); write_npy(d + "z.npy", s.z, {nz});
  write_npy(d + "V.npy", V, {nB, nz}); write_npy(d + "q.npy", todbl(sol.q), {nB, nz}); write_npy(d + "S.npy", todbl(sol.S), {nB, nz});
  write_npy(d + "c.npy", todbl(sol.c), {nB, nz}); write_npy(d + "d.npy", todbl(sol.d), {nB, nz}); write_npy(d + "g.npy", todbl(sol.g), {nB, nz});
  write_npy(d + "which.npy", std::vector<double>(sol.which.begin(), sol.which.end()), {nB, nz});
  write_npy(d + "frozen.npy", std::vector<double>(sol.frozen.begin(), sol.frozen.end()), {nB, nz});
  if (nrec > 0) {
    write_npy(d + "irf.npy", irf, {8, 8, (long)nrec});
    std::vector<double> t(nrec); for (int i = 0; i < nrec; ++i) t[i] = i * o.irf_dt;
    write_npy(d + "irf_t.npy", t, {(long)nrec});
  }
  FILE* f = fopen((d + "meta.json").c_str(), "w");
  fprintf(f, "{\n \"backend\": \"%s\", \"fp\": %d, \"nB\": %d, \"nz\": %d, \"threads\": %d,\n", o.backend.c_str(), o.fp, s.nB, s.nz, omp_get_max_threads());
  fprintf(f, " \"iters\": %d, \"converged\": %s, \"resV\": %.6e, \"resq\": %.6e, \"n_frozen\": %d, \"passes\": %ld, \"unconverged_sweeps\": %d, \"ergodic_res\": %.6e, \"stalled\": %s, \"maxpass\": %d, \"tolV\": %g, \"tolq\": %g, \"t_policy\": %.4f, \"t_sweep\": %.4f,\n",
          sol.iters, sol.converged ? "true" : "false", sol.resV, sol.resq, sol.n_frozen, sol.passes, sol.unconverged_solves, sol.ergodic_res, sol.stalled ? "true" : "false", o.maxpass, o.tolV, o.tolq, sol.t_policy, sol.t_sweep);
  fprintf(f, " \"accel_on\": %g, \"accel_grow\": %g, \"accel_dtmax\": %g, \"sim_layout\": %d, \"seg_years\": %d, \"burn_seg\": %d, \"irf_iters\": %d,\n",
          o.accel_on, o.accel_grow, o.accel_dtmax, o.sim_layout, o.seg_years, o.burn_seg, irf_iters_default(o, o.fp));
  fprintf(f, " \"times\": {");
  for (size_t i = 0; i < tm.t.size(); ++i) fprintf(f, "%s\"%s\": %.6f", i ? ", " : "", tm.t[i].first.c_str(), tm.t[i].second);
  fprintf(f, "},\n");
  auto dump = [&](const char* name, const std::map<std::string, double>& m, bool last) {
    fprintf(f, " \"%s\": {", name); size_t i = 0;
    for (auto& kv : m) fprintf(f, "%s\"%s\": %.10g", i++ ? ", " : "", kv.first.c_str(), kv.second);
    fprintf(f, "}%s\n", last ? "" : ",");
  };
  dump("ergodic", M.E, false); dump("sim_mean", M.Sm, false); dump("sim_sd", M.Ssd, false); dump("moments", M.M, false);
  fprintf(f, " \"loss\": %.10g\n}\n", M.loss);
  fclose(f);
  if (o.verbose) {
    printf("times:"); for (auto& kv : tm.t) printf(" %s %.4f", kv.first.c_str(), kv.second); printf("\n");
    printf("loss %.4f; ergodic residual %.2e\n", M.loss, sol.ergodic_res);
  }
}
