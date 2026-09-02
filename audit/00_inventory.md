# Phase 0 — Inventory (read-only audit, no code modified)

Audited at commit `530b48c` ("First working SEE solution") + uncommitted working-tree changes to
`main.py`, `atoms.py`, `formal_solver.py`, `config_H_Ca_Mg_Na.json`, `.gitignore`.
All line numbers refer to the working tree as of this audit.

---

## 1. Module map and call graph

```
main.py  (script, no functions; config hardcoded at main.py:14 -> 'config_H_Ca_Mg_Na.json')
│
├── constants.py                       CGS constants (kB, h, m_e, c, m_u, q_e, eV, a0, E_Ryd, ...)
│
├── atmosphere.py
│   ├── get_angular_quadrature_1D()    L8-28    Gauss-Legendre on [-1,1]; odd n bumped to even
│   ├── Atmosphere.from_dict()         L40-53   zgrid km->cm; T, pg, pel from JSON
│   │   └── __post_init__()            L60-84
│   │       ├── chemeq.compute_background_eos()      92-element LTE EOS -> ne, nh  (pel IGNORED)
│   │       └── chemeq.compute_background_species()  species/U for Wittmann background opacities
│   └── compute_lte_populations()      L86-175  Saha-Boltzmann per stage w/ chemeq partition fns
│       └── chemeq.get_partition_functions()
│
├── atoms.py
│   ├── MultiLevelAtom.from_dict()     L379-392 parses levels/lines/continua/collisions
│   │   └── __post_init__()            L325-377 cm^-1 -> erg; SORTS levels by energy and remaps
│   │                                            indices; f -> Aul,Bul,Blu (L370-375)
│   ├── Continuum.from_dict()          L162-170 ExplicitContinuum | HydrogenicContinuum
│   │   ├── ExplicitContinuum.alpha()  L212-222 linear interp in lambda, 0 outside table/edge
│   │   └── HydrogenicContinuum.alpha()L248-260 alpha0*(gbf/gbf0)*(lam/lam_edge)^3, Seaton gaunt
│   ├── get_barklem_cross_section()    L65-126  ABO tables (atomic_data.py) + spline in neff
│   ├── create_frequency_grid()        L415-543 union of per-line RH-style q-grids (built in
│   │                                            wavelength space) + per-continuum grids + 500 nm
│   │                                            point; GLOBAL trapezoid weights (L538-541);
│   │                                            line.max_delta_nu set at L493
│   ├── solve_SEE()                    L546-675 loop over depths; local (populations <-> ne)
│   │                                            iteration; ne update via delta-charge (L621-643)
│   └── solve_atom()                   L678-830 rate-matrix assembly + conservation row + solve
│
├── formal_solver.py
│   ├── plank()                        L237-260 B_nu(T), expm1-stable
│   ├── voigt()                        L262-303 Hui-style rational approx of w(z); Python loop
│   ├── get_RT_coefficients()          L307-395 line chi/eta (Voigt, truncate+renorm) + NLTE bf
│   │                                            chi/eta + background
│   ├── add_background_opacity()       L398-456 Wittmann/Mihalas suite: H bf/ff, H-, H2+, He I/II,
│   │                                            He- ff, cool metals (C,Mg,Al,Si,Fe),
│   │                                            luke metals (N,O,MgII,SiII,CaII), Thomson,
│   │                                            Rayleigh H/He/H2.  eta = kappa*B + sigma*J (L455)
│   ├── add_background_opacity_old()   L666-723 DEAD CODE (FIRTEZ cont_opacity.f90 port), unused
│   ├── formal_solution()              L912-937 linear short characteristics, Lambda* = psi_o
│   └── psi_lin()                      L939-963 psi_m/psi_o with Taylor branch for dtau <= 0.1
│
├── chemeq.py
│   ├── XI/XII/ABUND/MATOM tables      92 elements; for H, XI = 0.754 eV (H-!), XII = 13.595 eV
│   ├── get_partition_functions(Z,T)   L66-416  FIRTEZ atom_database.f90 3-stage fits.
│   │                                            NOTE: for Z=1 the triplet is (H-, H I, H II)
│   ├── compute_background_eos()       L421-483 iterative 92-element charge balance from (T, pg)
│   └── compute_background_species()   L522-670 Wittmann species densities / U
│
├── atomic_data.py                     Barklem/ABO sigma & alpha tables (sp, pd, df) only
└── debug_functions.py                 print_eq_system() only (diagnostics)
```

### Main-script flow (`main.py`)

1. L14-34: load config (hardcoded filename — **CLI args ignored**), create timestamped output dir,
   copy sources.
2. L38: angular quadrature (n_gaus=5 → bumped to 6 GL nodes on [-1,1]).
3. L47: `Atmosphere.from_dict` → runs full 92-element EOS; `pel` from JSON is read but unused.
4. L50-55: parse atoms; `populations = lte_populations = compute_lte_populations(...)`;
   `compute_doppler_widths` (fixed for the whole run).
5. L58-63: global frequency grid + global trapezoid weights.
6. L65-67: `atmosphere.J_nu` initialized to `B_nu(T)` (used only for background-scattering
   emissivity in iteration 1; overwritten at L312 each iteration).
7. L70-72: `hnu_grid` with `hnu_grid[hnu_grid==0]=1e-100` guard — **verified: the grid contains
   no zero frequency; the guard is vestigial** (nu range 7.40e13 – 1.32e16 Hz).
8. L80-90: per-continuum cross-sections interpolated once onto the global grid;
   `lte_ratios_photoionization[:, i_cont] = n_l*/n_u*` computed ONCE from `lte_populations`
   (L85-86) and never updated, even though `atmosphere.ne` is updated inside `solve_SEE`.
9. L94-193: per-line broadening constants precomputed (Unsold C6, Barklem, quadratic Stark C4,
   linear Stark factor).
10. L197-446: **Λ-iteration loop** (details below).
11. L452-457: departure coefficients b = n / n_LTE.
12. L468-506: final μ=1 formal solution, bottom BC `I = B(T[0])` (L474), τ bookkeeping
    (L476: `tau_depth[0,:]` set to a NONZERO value under a comment claiming zero).
13. L510-532: save .npy outputs; L534-715: diagnostic plots
    (L534 uses `configuration.get("debug", False)` while L40 uses `eval(configuration["debug"])`).

### Inside the Λ loop (what is recomputed per iteration)

| Quantity | Recomputed? | Where |
|---|---|---|
| `atom.Js`, `atom.Lambda_star_bar`, photo/recomb rates | reset & rebuilt each iteration | main.py:204-208 |
| `emis, abs` (all opacities incl. background) | per depth **per ray** (6× redundant) | main.py:229,243 |
| line damping / Voigt / truncation / renorm | per depth per ray per line, using **current** ne and current NLTE H-ground population `nHGround` | main.py:259-290 |
| `J_grid` (full-spectrum J) | rebuilt each iteration; assigned to `atmosphere.J_nu` **after** the ray loop (scattering emissivity therefore uses previous iteration's J — lagged Λ-iteration on scattering) | main.py:307,312 |
| bf photoionization/recombination rates | rebuilt from new `J_grid` each iteration | main.py:315-334 |
| `atmosphere.ne` | updated inside `solve_SEE` via delta-charge vs LTE baseline, damped 0.5, floored at `1e-6*nh` | atoms.py:621-643 |
| `atom.populations` | updated by `solve_atom` per depth (MALI-preconditioned rates) | atoms.py:611-615 |
| `lte_populations`, `lte_ratios_photoionization`, `bg_species`, `doppler_widths`, broadening constants | **fixed** (computed once, at `ne_bg`) | — |
| Convergence | `max |Δn|/avg` over populations **and** ne < `max_tolerance` (0.01) | atoms.py:654-675, main.py:444 |

No Ng acceleration. No residual (‖P·n‖) or J̄-based convergence criterion.

---

## 2. Physical arrays: name, shape, units, ordering, first assignment

Depth convention: **index 0 = BOTTOM** (z=0 km, T=9400 K), last index = TOP (z=2342 km,
T=1e5 K). `zgrid` is validated strictly increasing (atmosphere.py:69-70). No velocity field
exists anywhere (Atmosphere has no v_los attribute; profiles have no μ·v shift).

| Array | Shape | Units | First assigned |
|---|---|---|---|
| `atmosphere.zgrid` | (Nz=82) | cm (config in km ×1e5) | atmosphere.py:49 |
| `atmosphere.temp` | (Nz) | K | atmosphere.py:50 |
| `atmosphere.pg`, `pel` | (Nz) | dyn cm⁻² (`pel` parsed, then **ignored**) | atmosphere.py:51-52 |
| `atmosphere.ne`, `ne_bg` | (Nz) | cm⁻³ (from EOS, not from pel) | atmosphere.py:76-77 |
| `atmosphere.nh` | (Nz) | cm⁻³ total hydrogen nuclei | atmosphere.py:78 |
| `atmosphere.he_abund` | scalar | number ratio N_He/N_H = 0.1 | atmosphere.py:80 |
| `atmosphere.bg_species[...]` | (Nz) each | cm⁻³ (mostly n/U(T)) | chemeq.py:565 |
| `atmosphere.J_nu` | (Nz, Nν) | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ | main.py:65 |
| `frequency_grid` | (Nν=936) | Hz, ascending | atoms.py:498 |
| `weigths_freq_grid` | (Nν) | Hz (global trapezoid) | atoms.py:538-541 |
| `level.energy` | scalar | erg (config cm⁻¹ × hc) | atoms.py:332 |
| `line.nu0`, `max_delta_nu` | scalar | Hz | atoms.py:361, 493 |
| `line.lambda0` | scalar | cm | atoms.py:363 |
| `line.Aul` | scalar | s⁻¹ | atoms.py:372 |
| `line.Bul`, `Blu` | scalar | per (erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹), i.e. B·J̄ in s⁻¹ | atoms.py:374-375 |
| `atom.populations`, `lte_populations` | (Nz, Nlev) | cm⁻³ | main.py:52-53 |
| `atom.Js` | (Nz, Nlines) | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ (profile-weighted J̄) | main.py:54 |
| `atom.Lambda_star_bar` | (Nz, Nlines) | dimensionless | main.py:206 |
| `atom.doppler_widths` | (Nz, Nlines) | Hz (Δν_D = ν0/c·√(2kT/m + ξ²)) | atoms.py:413 |
| `atom.photoionization_alphas` | (Ncont, Nν) | cm² (config m² ×1e4) | main.py:81,90 |
| `atom.lte_ratios_photoionization` | (Nz, Ncont) | dimensionless (n_l*/n_u*) | main.py:82,85 |
| `atom.photoionization_rates`, `recombination_rates` | (Nz, Ncont) | s⁻¹ | main.py:333-334 |
| `emis` (η) | (Nν) per depth | erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹ | formal_solver.py:311 |
| `abs` (χ) | (Nν) per depth | cm⁻¹ | formal_solver.py:312 |
| `I_o`, `I_m` | (Nν) | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ | main.py:218/222 |
| `J_grid` | (Nz, Nν) | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ | main.py:210 |
| `tau_depth` | (Nz, Nν) | dimensionless (cumulative from bottom) | main.py:470 |
| collisions `rates` (CE/CI) | table | m³ s⁻¹ K^(-1/2) (RH convention; ×1e6 → cm³ at use) | atoms.py:721,726 |
| collisions `rates` (Omega) | table | dimensionless Υ(T) | atoms.py:713-717 |

---

## 3. The atomic models as actually parsed (`config_H_Ca_Mg_Na.json`)

Despite the file name, only **H (6 levels)** and **Ca II (6 levels)** are present. Both are
both active NLTE **as originally written** — there was no `is_active` handling anywhere in
the parser or main loop, so the flag written by `tests/compare_snapi/create_snapi_config.py`
was silently ignored. **This has since been implemented** (see FINDINGS F-017 commit):
`MultiLevelAtom.is_active` is parsed, passive atoms are held at their LTE populations and
skipped by `solve_SEE` while still contributing line and bound-free opacity. It was needed
because the SNAPI benchmark run held hydrogen in LTE.

Level parsing notes:
- Energies in cm⁻¹ → erg at atoms.py:332. Levels **re-sorted by energy** and all transition
  indices remapped (atoms.py:336-348). For these models input order is already sorted.
- H: 1s(g=2), 2(g=8), 3(g=18), 4(g=32), 5(g=50), H II (g=1, ionization=1).
  (Merged-l levels; g=2n².)
- Ca II: 4s(g=2), 3d 3/2(g=4), 3d 5/2(g=6), 4p 1/2(g=2), 4p 3/2(g=4), Ca III (g=1, ionization=2).

Lines (f-values → Einstein coefficients at atoms.py:370-375; **verified numerically**:
`A = 8π²e²ν²/(m_e c³)·(g_l/g_u)·f` reproduces NIST values, e.g. A(Hα)=4.407e7, A(Lyα)=4.696e8,
A(CaII K)=1.466e8, A(854.2)=9.92e6; Einstein relations `A=(2hν³/c²)B_ul`, `g_lB_lu=g_uB_ul` hold
to machine precision for all 15 lines):

| Atom | Line | λ₀ (vac, nm) | f | type in file | γ_rad (s⁻¹) | Elastic broadening |
|---|---|---|---|---|---|---|
| H | 1-2 (Lyα) | 121.57 | 0.4162 | "PRD" (treated CRD) | 4.70e8 | Unsold + quadratic Stark + linear Stark |
| H | 1-3 (Lyβ) | 102.57 | 0.0791 | "PRD" (CRD) | 9.98e7 | same |
| H | 1-4, 1-5 | 97.3, 95.0 | ... | CRD | ... | same |
| H | 2-3 (Hα) | 656.47 | 0.6407 | CRD | 9.98e7 | same |
| H | 2-4 (Hβ), 2-5 (Hγ), 3-4 (Paα), 3-5, 4-5 (Brα) | ... | ... | CRD | ... | same |
| Ca II | 1-4 (H) | 396.96 | 0.3412 | "PRD" (CRD) | 1.48e8 | Unsold(1.5,1.0) + quad Stark |
| Ca II | 1-5 (K) | 393.48 | 0.6807 | "PRD" (CRD) | 1.50e8 | same |
| Ca II | 2-4 (866.5), 2-5 (850.0), 3-5 (854.4) | IR triplet | ... | CRD | ... | same |

- The `type` field ("PRD"/"CRD") is parsed but **never used**; everything is CRD.
- `quadrature` (N_lambda, q_core, q_wing) drives the RH-style per-line q-grid
  (atoms.py:457-493): N_half=(N_λ+1)//2 points, built **symmetric in wavelength** around λ₀
  with a fixed characteristic Doppler width λ₀·v_turb/c (v_turb = 3 km/s from config),
  mirroring Lightweaver's default `vMicroChar = 3 km/s`.
- `line.max_delta_nu` = max |ν−ν₀| of the line's own grid (atoms.py:493); used to truncate the
  Voigt profile on the global grid (main.py:287-288, formal_solver.py:352-353) before
  **numerical renormalization** against the global weights (main.py:290, formal_solver.py:355).

Continua:
- H: 5 × `HydrogenicContinuum` (Lyman/Balmer/... edges), `alpha0` in m² → cm²,
  α(λ)=α₀·(g_bf(λ)/g_bf(edge))·(λ/λ_edge)³, Seaton (1960) Gaunt factor (atoms.py:262-288),
  zero below `minWavelength` and above the edge.
- Ca II: 5 × `ExplicitContinuum`, tabulated (λ[nm], σ[m²→cm²]), linear interp in λ, zero
  outside table and beyond edge; negative interpolants clamped to 0 (atoms.py:221).
- The bf grids contribute their wavelength points to the global frequency grid
  (atoms.py:447-450), plus the λ edge appended when missing (atoms.py:201-210).

Collisions (atoms.py:694-748):
- H: `CE` (10 pairs) and `CI` (5), RH units (m³ K^-1/2 s^-1 → ×1e6 to cm³).
  CE gives the **downward** rate: C_ji = rate·ne·(g_i/g_j)·√T; upward from detailed balance
  with exp(−ΔE/kT). CI: upward C_ij = rate·ne·√T·exp(−ΔE/kT); downward via the LTE
  population ratio (from stored `lte_populations` → frozen at ne_bg).
- Ca II: `Omega` (10 pairs; dimensionless Υ, C_ji = 8.629e-6·ne·Υ/(g_j√T)) and `CI` (5).
- Temperature interpolation: `np.interp` — **linear in T, linear in rate, clamped at table
  edges** (atoms.py:705). Tables span 3000–1e6 K? (H tables start at 3000 K; top of atmosphere
  is 1e5 K — inside table range; verified edges clamp).
- Parser silently skips unknown types with a printed warning (atoms.py:743-745).
  Supported: Omega, CE, CI, CH, CP. Note `CP` multiplies by **nh (total hydrogen)**, not the
  proton density (atoms.py:741) — unused by current configs but wrong as written; `CH` also
  multiplies by total nh rather than neutral-H density.

Partition functions: `irwin_coefficients` parsed from JSON (empty in this config) but the Irwin
path is commented out (chemeq.py:673-691); `compute_lte_populations` uses
`chemeq.get_partition_functions(atom.Z, T)` instead — see the **critical H mapping issue** in
§5 below.

---

## 4. The atmosphere as actually used

- FAL-C-like 82-point stratification in the main config (bottom z=0/T=9400 K, top
  z=2342 km/T=1e5 K); the SNAPI comparison generates a separate 128-point FALC config.
- `zgrid` increasing **upward**; index 0 = bottom.
- `ne`: **solved**, twice over. (1) Background 92-element LTE EOS from (T, pg) ignoring the
  config's `pel` (atmosphere.py:74-78); (2) then updated each Λ-iteration inside `solve_SEE`
  via Δcharge of the active atoms added to `ne_bg`, with 0.5 damping and a `1e-6·nh` floor
  (atoms.py:625-643). NLTE populations therefore do feed back into ne, but
  `lte_populations`, `lte_ratios_photoionization` and `bg_species` are never re-evaluated at
  the updated ne (inconsistency, to be quantified in Phase 2).
- No velocity field, no μ-dependence in profiles: correct for static FALC; a missing feature
  (class C) for anything else.
- Microturbulence: single constant `turbulent_velocity` = 3e5 cm/s (FALC's depth-dependent
  v_turb column is ignored in the SNAPI config generator, which sets it to **0.0** —
  a setup mismatch to resolve in Phase 4).

---

## 5. Verified facts from read-only diagnostics (basis for Phase 2 findings)

1. **Angular quadrature is correct in convention**: `leggauss(6)` on [−1,1], Σw = 2, no μ=0
   node; `J += 0.5·w·I` is the proper ∫dμ/2. Emergent-ray pick at main.py:309 selects
   μ=0.932 (largest node), not μ=1 — used only for per-iteration debug plots; the final
   spectrum is a separate true μ=1 pass.
2. **`voigt()` approximates H(a,v)=Re[w(v+ia)]** to ≲5e-5 relative (checked against
   `scipy.special.wofz` for a ∈ [1e-4, 1]); Gaussian and Lorentzian limits verified.
3. **`psi_lin` has no small-Δτ cancellation problem**: the Taylor branch (Δτ≤0.1) matches
   reference values from 1e-10 to 1e2, and ψ_m+ψ_o+e^(−Δτ) = 1 to machine precision.
   The seed concern about `expm1` is already handled by the series expansion.
4. **Einstein relations and f→A conversion are exact** (see §3).
5. **The frequency grid contains no zero or negative frequency** (min 7.40e13 Hz);
   the `hnu_grid==0` guard at main.py:71 is dead.
6. **Analytic profile normalization on the global grid fails at the 0.2–1.4% level**
   (∫φdν = 1.002 for Lyα, 1.014 for Ca II H at a=1e-3) — the numerical renormalization at
   main.py:290 is absorbing a real quadrature/grid error (wing trapezoid + gap-edge weights).
   To be dissected in Phase 2.3 (this is the author's "why does this matter so much" TODO).
7. **`get_partition_functions(1,T)` returns (1.0, 2.0, 1.0)** = (U(H⁻), U(H I), U(H II)) —
   the FIRTEZ Z=1 triplet is stage-shifted (H⁻/H/H⁺), consistent with `XI[0]=0.754 eV` being
   the H⁻ binding energy and with its use in `compute_background_eos`.
   But `compute_lte_populations` maps `U_t = {0: UI, 1: UII, 2: UIII}` (atmosphere.py:148-149),
   i.e. assigns **U(H⁻)=1 to H I and U(H I)=2 to H II**. Measured consequence: Σᵢnᵢ/N_total
   for H ranges from **2.000** (cool layers) to **0.004** (1e5 K top, where the T-corrected
   U(H I)≈249 is applied to the proton level). Level-to-level LTE *ratios* (Boltzmann/Saha)
   remain exactly correct because U cancels in ratios — so rates built from ratios are clean,
   but absolute `lte_populations`, departure coefficients, the Δcharge ne update baseline
   (atoms.py:594) and first-iteration opacities are wrong by depth-dependent factors of
   0.004–2. Candidate finding F-001 (CRITICAL, class A).
8. `run_full_comparison.py` cannot run what it claims: it passes a config path that `main.py`
   ignores (main.py:14), the generated config drops the `pel` key that `Atmosphere.from_dict`
   requires (KeyError), and sets an `is_active: False` flag for H that nothing reads.
   The existing benchmark comparison must be treated as unvalidated until Phase 4 re-runs it.

## 6. Benchmark data present (`tests/compare_snapi`)

- `snapi_firtez_comparison_data.npz`: `atm_lvl_pops` (97 iter × 128 depth × 13 = 6 H levels +
  6 Ca II levels + ne), `height_grid_cm` (128), `debug_J/L/norm` (97×128×15 transitions;
  J in **erg cm⁻² s⁻¹ cm⁻¹** — per wavelength!), `rate_matrix` (97×128×6×6),
  `falc_model` (128×12: log τ500, z cm, T, pg, pel, v_turb, ...), `synth_spectrum`
  (501×5: λ cm, I, Q, U, V).
- `calcium_ir_rh.cfg`: the SNAPI Ca model — 5 Ca II levels + continuum, ABUND=2.1877624e-06
  (config uses **2.138e-06**: a 2.3% setup mismatch), Traving partition functions, tabulated bf.
- SNAPI solved **H and Ca II in NLTE simultaneously** (13 population rows), with FALC's
  depth-dependent v_turb.
