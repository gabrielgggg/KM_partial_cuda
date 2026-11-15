// km.cu -- performance-oriented solver for the diffusion model of Kim and Mihalache, "Debt Capacity and the
// Anatomy of Sovereign Crises": equilibrium (V, q, policies), ergodic distribution, moments (exact ergodic +
// annual-accounting simulation), and impulse responses. Two back ends share the model code in model.h:
//   --backend gpu   everything on the GPU (CUDA), host only steers the iteration
//   --backend cpu   everything on the CPU (C++/OpenMP, all cores)
// and the float type is a template parameter: --fp 64 (double) or --fp 32 (float).
// Results are written as .npy files plus meta.json in --out DIR; plotting is done in Python (tools/).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include "model.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ------------------------------------------------------------------------------------------- options
struct Opt {
  std::string backend = "gpu", out = "out", init = "init.bin", params = "params.txt";
  int fp = 64, nB = 201, nz = 75, maxit = 5000, maxpass = 60, n_golden = 28;
  double Bmax = 1.0, nsd = 2.5, dt0 = 0.015, dt_grow = 1.02, dt_max = 0.5, sticky = 1e-6;
  int freeze_flips = 3, freeze_window = 30, extrap = 1, stall_window = 150; double tol32 = 64, relax = 1.0; int chunk_rows = -1, irf_all_rows = 400;   // rows per chunk of the block sweeps; -1: automatic, 0: one chunk per system
  double tolV = 1e-7, tolq = 1e-8, sweep_tol = 1e-13;
  // acceleration beyond the paper's schedule: once both residuals are below accel_on, the step grows by
  // accel_grow per step up to accel_dtmax (0 = off: the paper's schedule)
  double accel_on = 0.0, accel_grow = 1.0, accel_dtmax = 0.5;
  // simulation
  int n_paths = 16, years = 25000, spy = 100, seg_years = 125, burn_seg = 25, burn_paper = 1000, sim_layout = 1;  // 1: segmented, 0: paper
  unsigned seed = 1;
  // impulse responses
  double irf_T = 21.0, irf_dt = 0.01, irf_shock = -0.10; int irf_iters = 0;
  int do_sim = 1, do_irf = 1, verbose = 1, write = 1;
};

// ------------------------------------------------------------------------------------------- npy writer
static void write_npy(const std::string& path, const std::vector<double>& v, std::vector<long> shape) {
  FILE* f = fopen(path.c_str(), "wb");
  std::string sh = "(";
  for (size_t i = 0; i < shape.size(); ++i) sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
  sh += ")";
  std::string hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': " + sh + ", }";
  while ((10 + hdr.size() + 1) % 64) hdr += ' ';
  hdr += '\n';
  unsigned short hl = (unsigned short)hdr.size();
  fwrite("\x93NUMPY\x01\x00", 1, 8, f); fwrite(&hl, 2, 1, f); fwrite(hdr.data(), 1, hdr.size(), f);
  fwrite(v.data(), sizeof(double), v.size(), f); fclose(f);
}
template <typename T> static std::vector<double> todbl(const std::vector<T>& a) { return std::vector<double>(a.begin(), a.end()); }

// ------------------------------------------------------------------------------------------- problem setup
struct Raw { double x[9]; double qf, band, lband; };
static Raw read_params(const std::string& fn) {
  Raw r; FILE* f = fopen(fn.c_str(), "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); exit(1); }
  for (int i = 0; i < 9; ++i) if (fscanf(f, "%lf", &r.x[i]) != 1) exit(2);
  if (fscanf(f, "%lf %lf %lf", &r.qf, &r.band, &r.lband) != 3) exit(2);
  fclose(f); return r;
}

struct Setup {  // double-precision description of the discretized model
  int nB, nz, N;
  std::vector<double> B, z, y, Fz, hf, hb, zup, zdn;
  double mu, sigma, r, delta, lam, kappa, rho;
  Par<double> p;
};

static Setup make_setup(const Raw& R, const Opt& o) {
  Setup s; s.nB = o.nB; s.nz = o.nz; s.N = o.nB * o.nz;
  double g0 = R.x[0], g1 = R.x[1], g2 = R.x[2], zt = R.x[3], rho = R.x[4], kappa = R.x[5], delta = R.x[6], rho_z = R.x[7], sig = R.x[8];
  double mu = -log(rho_z), sd = sig / sqrt(1 - rho_z * rho_z), sigma = sd * sqrt(2 * mu);
  double r = log(1.04);
  s.mu = mu; s.sigma = sigma; s.r = r; s.delta = delta; s.lam = r; s.kappa = kappa; s.rho = rho;
  Par<double>& p = s.p;
  p.nu = 2; p.rho = rho; p.delta = delta; p.kappa = kappa; p.r = r; p.lam = r; p.g0 = g0; p.g1 = g1; p.g2 = g2; p.zt = zt;
  p.cmf = 0.05; p.Scap = 1.0; p.Fmin = 0.10; p.qf = R.qf; p.qband = R.band; p.lband = R.lband; p.closed_d = 1; p.n_golden = o.n_golden;
  s.B.resize(s.nB); s.z.resize(s.nz);
  for (int i = 0; i < s.nB; ++i) s.B[i] = 0.0 + (o.Bmax - 0.0) * i / (s.nB - 1);
  double zmin = -o.nsd * sd, zmax = o.nsd * sd;
  for (int j = 0; j < s.nz; ++j) s.z[j] = zmin + (zmax - zmin) * j / (s.nz - 1);
  s.y.resize(s.nz); s.Fz.resize(s.nz);
  for (int j = 0; j < s.nz; ++j) {
    s.y[j] = exp(s.z[j]);
    double F = 1.0;
    if (s.z[j] >= zt) { double v = 1.0 - (exp(s.z[j]) - exp(zt)) * g2; F = v > p.Fmin ? v : p.Fmin; }
    s.Fz[j] = F;
  }
  s.hf.resize(s.nB); s.hb.resize(s.nB);
  for (int i = 0; i < s.nB - 1; ++i) s.hf[i] = s.B[i + 1] - s.B[i];
  s.hf[s.nB - 1] = s.hf[s.nB - 2];
  for (int i = 1; i < s.nB; ++i) s.hb[i] = s.B[i] - s.B[i - 1];
  s.hb[0] = s.hb[1];
  s.zup.assign(s.nz, 0.0); s.zdn.assign(s.nz, 0.0);
  double dz = s.z[1] - s.z[0], diff = 0.5 * sigma * sigma / (dz * dz);
  for (int j = 0; j < s.nz; ++j) {
    double m = -mu * s.z[j];
    if (j < s.nz - 1) s.zup[j] = std::max(m, 0.0) / dz + diff;
    if (j > 0) s.zdn[j] = std::max(-m, 0.0) / dz + diff;
  }
  return s;
}

// bilinear (clamped) interpolation of a saved (V, q) onto the grid; q clipped to [1e-6, 1]
static void read_init(const std::string& fn, const Setup& s, std::vector<double>& V, std::vector<double>& q) {
  FILE* f = fopen(fn.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); exit(1); }
  int n[2]; if (fread(n, 4, 2, f) != 2) exit(3);
  int nB0 = n[0], nz0 = n[1];
  std::vector<double> B0(nB0), z0(nz0), V0((size_t)nB0 * nz0), q0((size_t)nB0 * nz0);
  if (fread(B0.data(), 8, nB0, f) != (size_t)nB0 || fread(z0.data(), 8, nz0, f) != (size_t)nz0 ||
      fread(V0.data(), 8, V0.size(), f) != V0.size() || fread(q0.data(), 8, q0.size(), f) != q0.size()) exit(3);
  fclose(f);
  auto loc = [](const std::vector<double>& g, double x, int& i, double& w) {
    int n = (int)g.size();
    if (x <= g[0]) { i = 0; w = 0; return; }
    if (x >= g[n - 1]) { i = n - 2; w = 1; return; }
    i = (int)(std::upper_bound(g.begin(), g.end(), x) - g.begin()) - 1;
    if (i > n - 2) i = n - 2;
    w = (x - g[i]) / (g[i + 1] - g[i]);
  };
  V.resize(s.N); q.resize(s.N);
  for (int i = 0; i < s.nB; ++i) {
    int a; double wa; loc(B0, s.B[i], a, wa);
    for (int j = 0; j < s.nz; ++j) {
      int b; double wb; loc(z0, s.z[j], b, wb);
      auto I = [&](const std::vector<double>& F) {
        return (1 - wa) * (1 - wb) * F[(size_t)a * nz0 + b] + wa * (1 - wb) * F[(size_t)(a + 1) * nz0 + b] +
               (1 - wa) * wb * F[(size_t)a * nz0 + b + 1] + wa * wb * F[(size_t)(a + 1) * nz0 + b + 1];
      };
      V[i * s.nz + j] = I(V0);
      q[i * s.nz + j] = std::min(1.0, std::max(1e-6, I(q0)));
    }
  }
}

template <typename T> static Par<T> castp(const Par<double>& d) {
  Par<T> p;
  p.nu = (T)d.nu; p.rho = (T)d.rho; p.delta = (T)d.delta; p.kappa = (T)d.kappa; p.r = (T)d.r; p.lam = (T)d.lam;
  p.g0 = (T)d.g0; p.g1 = (T)d.g1; p.g2 = (T)d.g2; p.zt = (T)d.zt; p.cmf = (T)d.cmf; p.Scap = (T)d.Scap; p.Fmin = (T)d.Fmin;
  p.qf = (T)d.qf; p.qband = (T)d.qband; p.lband = (T)d.lband; p.closed_d = d.closed_d; p.n_golden = d.n_golden;
  return p;
}
template <typename T> static std::vector<T> castv(const std::vector<double>& v) { return std::vector<T>(v.begin(), v.end()); }

// ------------------------------------------------------------------------------------------- solution container
template <typename T> struct Sol {
  std::vector<T> W, q, S, c, d, u, g;
  std::vector<int8_t> which; std::vector<uint8_t> frozen;
  double Vref = 0, resV = 0, resq = 0; int iters = 0, n_frozen = 0; long passes = 0; bool converged = false;
  int unconverged_solves = 0; double ergodic_res = 0; double t_policy = 0, t_sweep = 0; int stalled_solves = 0; bool stalled = false;
};

struct Timer { std::vector<std::pair<std::string, double>> t; void add(const std::string& k, double v) { t.push_back({k, v}); } };

#include "cpu_backend.h"
#include "gpu_backend.cuh"
#include "post.h"

template <typename T> static int run(const Opt& o) {
  Raw R = read_params(o.params);
  Setup s = make_setup(R, o);
  std::vector<double> V0, q0;
  read_init(o.init, s, V0, q0);
  Timer tm;
  Sol<T> sol;
  double t0 = now();
  if (o.backend == "gpu") gpu::warmup();
  tm.add("init", now() - t0);
  if (o.verbose) printf("backend %s, fp%d, grid %d x %d, threads %d\n", o.backend.c_str(), o.fp, s.nB, s.nz, omp_get_max_threads());
  t0 = now();
  if (o.backend == "gpu") gpu::solve<T>(s, o, V0, q0, sol); else cpu::solve<T>(s, o, V0, q0, sol);
  tm.add("solve", now() - t0);
  if (o.verbose) printf("solve: %d steps, resV %.2e resq %.2e, frozen %d, sweep passes %ld, unconverged sweeps %d, %.3fs\n",
                        sol.iters, sol.resV, sol.resq, sol.n_frozen, sol.passes, sol.unconverged_solves, tm.t.back().second);
  if (o.verbose && sol.t_policy > 0) printf("  policy %.3fs, sweeps %.3fs\n", sol.t_policy, sol.t_sweep);
  t0 = now();
  ergodic<T>(s, sol);
  tm.add("ergodic", now() - t0);
  t0 = now();
  Moments M = ergodic_moments<T>(s, sol);
  tm.add("ergodic_moments", now() - t0);
  std::vector<double> sims; int L = 0, P = 0, Y = 0;
  if (o.do_sim) {
    t0 = now();
    if (o.backend == "gpu") gpu::simulate<T>(s, o, sol, sims, P, Y, L); else cpu::simulate<T>(s, o, sol, sims, P, Y, L);
    tm.add("simulate", now() - t0);
    t0 = now();
    path_moments(sims, P, Y, L, M);
    tm.add("path_moments", now() - t0);
  }
  std::vector<double> irf; int nrec = 0;
  if (o.do_irf) {
    t0 = now();
    if (o.backend == "gpu") gpu::irf<T>(s, o, sol, irf, nrec); else cpu::irf<T>(s, o, sol, irf, nrec);
    tm.add("irf", now() - t0);
  }
  write_all<T>(s, o, sol, M, tm, irf, nrec);
  return 0;
}

int main(int argc, char** argv) {
  Opt o;
  for (int a = 1; a < argc; ++a) {
    std::string k = argv[a];
    auto nx = [&]() { if (a + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(1); } return std::string(argv[++a]); };
    if (k == "--backend") o.backend = nx(); else if (k == "--out") o.out = nx(); else if (k == "--init") o.init = nx();
    else if (k == "--params") o.params = nx(); else if (k == "--fp") o.fp = std::stoi(nx());
    else if (k == "--nB") o.nB = std::stoi(nx()); else if (k == "--nz") o.nz = std::stoi(nx());
    else if (k == "--maxit") o.maxit = std::stoi(nx()); else if (k == "--maxpass") o.maxpass = std::stoi(nx());
    else if (k == "--tolV") o.tolV = std::stod(nx()); else if (k == "--tolq") o.tolq = std::stod(nx());
    else if (k == "--sweep_tol") o.sweep_tol = std::stod(nx());
    else if (k == "--dt_max") o.dt_max = std::stod(nx()); else if (k == "--dt0") o.dt0 = std::stod(nx());
    else if (k == "--accel_on") o.accel_on = std::stod(nx()); else if (k == "--accel_grow") o.accel_grow = std::stod(nx());
    else if (k == "--accel_dtmax") o.accel_dtmax = std::stod(nx());
    else if (k == "--paths") o.n_paths = std::stoi(nx()); else if (k == "--years") o.years = std::stoi(nx());
    else if (k == "--seg_years") o.seg_years = std::stoi(nx()); else if (k == "--burn_seg") o.burn_seg = std::stoi(nx());
    else if (k == "--sim_layout") o.sim_layout = std::stoi(nx()); else if (k == "--seed") o.seed = (unsigned)std::stoul(nx());
    else if (k == "--extrap") o.extrap = std::stoi(nx());
    else if (k == "--chunk_rows") o.chunk_rows = std::stoi(nx());
    else if (k == "--n_golden") o.n_golden = std::stoi(nx());
    else if (k == "--relax") o.relax = std::stod(nx());
    else if (k == "--irf_all_rows") o.irf_all_rows = std::stoi(nx());
    else if (k == "--tol32") o.tol32 = std::stod(nx());
    else if (k == "--stall_window") o.stall_window = std::stoi(nx());
    else if (k == "--irf_iters") o.irf_iters = std::stoi(nx());
    else if (k == "--nosim") o.do_sim = 0; else if (k == "--noirf") o.do_irf = 0; else if (k == "--quiet") o.verbose = 0; else if (k == "--verbose") o.verbose = 2;
    else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
  }
  mkdir(o.out.c_str(), 0755);
  // flush subnormals to zero on every OpenMP thread (densities in the far tails underflow; avoids slow paths)
#pragma omp parallel
  { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON); }
  return o.fp == 32 ? run<float>(o) : run<double>(o);
}
