# Findings register — firtez-nlte audit

Sorted by severity. Class: **A** wrong physics · **B** correct physics, wrong numerics ·
**C** correct but non-standard.

Status legend: `PATCHED` (fix committed + regression test) · `RECORDED` (diagnosed, not
patched, reason given) · `REFUTED` (investigated, no defect — kept so the question is not
re-opened).

---

## F-001 — CRITICAL — Hydrogen partition functions are assigned to the wrong ionization stage

    Class:      A (wrong physics)
    Location:   atmosphere.py:148-149  (compute_lte_populations)
    Standard:   HM2014 §5.1 (Saha–Boltzmann); chemeq.py's own tables and
                chemeq.compute_background_species:578-586, which explicitly documents
                U(H I)=2.0, U(H II)=1.0

**Correct:** for element Z the stage-s partition function must be U of that stage;
n_i = N_s (g_i/U_s) exp(−E_i^rel/k_BT), Σ_i n_i = A_elem n_H.

**Implemented:** `U_t = {0: UI, 1: UII, 2: UIII}` for every element. But
`get_partition_functions(1, T)` returns the FIRTEZ hydrogen triplet **(U(H⁻), U(H I), U(H II))
= (1.0, 2.0, 1.0)** — hydrogen's stages in that table are H⁻/H/H⁺, which is why
`XI[0] = 0.754 eV` is the H⁻ electron affinity and `XII[0] = 13.595 eV` is the H I ionization
potential. So H I is given U(H⁻)=1 instead of 2, and H II is given U(H I) instead of 1.

**Impact:** Σᵢnᵢ(H)/N_total = **2.000** at cool depths → **0.0040** at the 1e5 K top (where
the excited-state-corrected U(H I)=249 lands on the bare proton). Corrupted: every hydrogen
**departure coefficient** b_i = n/n_LTE (the headline deliverable) by factors 0.004–248; the
initial guess `populations = lte_populations`; and `lte_total_charge` (atoms.py:594), the
baseline of the Δcharge electron-density update, by up to 249× — which is precisely why the
`1e-6·n_H` electron floor at atoms.py:630-633 exists. **Not** corrupted: all level *ratios*
— `lte_ratios_photoionization`, the CI/CH detailed-balance ratios, the bf emissivity ratio —
because U cancels exactly in a ratio (proven analytically in 02_physics.md §2.5).

**Evidence:** Test C (`test_detailed_balance_*`). Forcing J_ν = B_ν, `solve_SEE` returns
populations whose *shape* matches Saha–Boltzmann to 1.1e-5 but whose *raw* ratio to
`lte_populations` is **248**. Ca II, whose stage mapping is accidentally correct, gives raw
7.1e-4 and shape 5.2e-8. The SEE is right; the LTE reference is wrong.

**Fix:** select the partition function by stage with a hydrogen-specific offset —
for Z=1, stage s ↦ the (s+1)-th entry of the triplet.

**Verified:** `test_lte_population_conservation` (Σᵢnᵢ/N_total = 1 to 1e-6 for both atoms,
all depths) and `test_detailed_balance_raw` (raw deviation now 1e-5, was 248).
Status: **PATCHED**.

---

## F-002 — CRITICAL — Global quadrature weights used for per-line profile integrals inflate ∫φdν by up to 13.3×

    Class:      B (correct physics, wrong numerics)
    Location:   atoms.py:538-541 (global weights); main.py:290,303,305;
                formal_solver.py:355
    Standard:   RH getlambda.c / Transition wlamb; Lightweaver
                Transition.compute_phi (per-transition `wphi` over the line's own subgrid)

**Correct:** ∫φ_ν dν = 1, with the quadrature confined to the transition's own wavelength
subgrid; J̄ = ∮(dΩ/4π)∫φ I dν over the same subgrid.

**Implemented:** `create_frequency_grid` returns one *global* trapezoid weight array over the
union grid, and that array is used for the per-line profile normalization, J̄ and Λ̄*. The
outermost point of each line's truncation window receives
`w = ½(ν[i+1] − ν[i−1])` where ν[i+1] is the next point of the **global** grid — i.e. the
weight straddles the empty gap separating this line's frequency block from the next block in
the spectrum.

**Impact:** measured ∫φdν over the global grid, natural damping:

| line | λ₀ (nm) | iz=0 | iz=41 | iz=81 (T=1e5 K) |
|---|---|---|---|---|
| H Lyα | 121.57 | 1.0020 | 1.0019 | 1.0031 |
| H 1→5 | 94.98 | 1.1020 | 1.1060 | 1.0781 |
| H Paα | 1875.63 | 1.0053 | 1.0052 | **1.7505** |
| H 3→5 | 1282.17 | 1.0612 | 1.0617 | **1.7813** |
| **H Brα** | **4052.29** | **1.1168** | **1.1121** | **13.3063** |
| Ca II H / K | 396.96 / 393.48 | 1.0140 | 1.0143 | 1.0117 |
| Ca II 866.5 | 866.45 | 1.0226 | 1.0225 | 1.0249 |

For Brα at iz=81 a *single* point — the window edge at v = 2.21 — carries a trapezoid weight
of **2.88e13 Hz** against a local grid spacing of 4.0e9 Hz, contributing **12.22 of the total
13.31 (92%)**. Since `voigt_norm = voigt_line / Σ(voigt_line·w)`, this
(a) deflates χ_line and η_line by up to 13×, (b) makes J̄ = Σwφ I / Σwφ a weighted average of
I dominated by the window-edge sample, so **J̄ for the isolated infrared hydrogen lines is
essentially the local continuum intensity rather than the line radiation field**, and
(c) deflates Λ̄*, degrading the MALI preconditioning and driving the defensive clamps at
atoms.py:760 and atoms.py:775. Ca II — the benchmark target — is deflated by a uniform
1.2–2.5%.

This is the answer to the author's TODO at main.py:286 ("why does this matter so much"): the
sensitivity to the truncation mask is **not** about damping-wing physics. Measured
Σ_outside(φw)/Σ_inside(φw) = 1e-9 … 9e-6 for all 15 lines, i.e. the discarded wing is
negligible. The mask matters because it *relocates the enormous edge weight*.

**Evidence:** `test_profile_normalization_analytic` (fails on the unpatched code with
∫φdν = 13.3), plus the per-point contribution table above.

**Fix:** compute per-line trapezoid weights restricted to the line's own truncation window
(half-intervals at the two ends), store them on the `Line` object once, and use them for the
profile normalization, J̄ and Λ̄*. Numerical renormalization is *retained* — it is what
Lightweaver's `wphi` does — but it now divides by a number that is genuinely ≈1.

**Verified:** `test_profile_normalization_analytic` (∫φdν = 1 ± 1e-6 for every line at every
depth, before renormalization); `test_lambda_star_bounded` (Λ̄* ≤ 1 without clipping).
Status: **PATCHED**.

---

## F-014 — HIGH — Angular quadrature integrates across the discontinuity of I(mu) at mu = 0

    Class:      B (correct physics, wrong numerics)
    Location:   atmosphere.py:8-28  (get_angular_quadrature_1D)
    Standard:   RH and Lightweaver both apply Gauss-Legendre per hemisphere;
                HM2014 §11.3 (angular quadrature for 1-D transfer)

**Correct:** J_nu = ½∫₋₁¹ I dμ. I(μ) is **discontinuous at μ = 0** — at the upper boundary
there is no incoming radiation, so I(μ<0) = 0 while I(μ>0) is finite. The integral must
therefore be split at μ = 0 and each hemisphere integrated separately.

**Implemented:** a single `np.polynomial.legendre.leggauss(n)` rule spanning [−1,1], placing
nodes straight across the jump. Gauss-Legendre across a discontinuity converges as O(1/N)
rather than spectrally.

**Impact:** measured error in J̄ — the profile-weighted mean intensity that feeds the
radiative rates — against a 64-ray reference, Ca II, at equal ray count:

| rays | full-range GL (before) | hemispheric GL (after) |
|---|---|---|
| 4 | 7.45e-02 | 4.13e-02 |
| 6 (**production**) | **5.54e-02** | **2.36e-02** |
| 8 | 4.49e-02 | 1.55e-02 |
| 16 | 2.57e-02 | 5.63e-03 |
| 24 | 1.87e-02 | 3.00e-03 |

A 2–5% error in J̄ propagates directly into the Ca II departure coefficients and the
emergent line cores — the benchmark's primary observables.

**Evidence:** `test_K3_hemispheric_quadrature_beats_full_range`, plus the self-convergence
study above. Note this was *invisible* to the isothermal-slab test (Test H still passes to
machine precision either way, because there I(μ) is constant over each hemisphere) — which
is exactly why the brief's warning that quadrature errors are "invisible in the LTE
continuum and lethal in the line cores" applies here, just not as a factor-of-2.

**Fix:** Gauss-Legendre on [0,1] mirrored into both hemispheres. Interface unchanged: same
node count, symmetric about zero, Σw = 2, no μ = 0 node — so `0.5·w·I` remains exactly
½∫₋₁¹I dμ and the `if ray > 0` direction test stays valid.

**Verified:** `test_H_isothermal_slab_gives_J_equals_B` still exact to 1e-12;
`test_K_angular_quadrature_converges`; `test_K3_hemispheric_quadrature_beats_full_range`.
Status: **PATCHED**.

**Not fixed — a configuration choice, deliberately left to the user.** `n_gaus = 5` (6 rays)
is too coarse *regardless of the rule*: even hemispheric, Ca II J̄ still moves **1.7%**
between 6 and 10 rays. **n_gaus ≥ 12 is recommended for quantitative work.** Raised in
`OPEN_QUESTIONS.md` as a parameter that must be agreed with SNAPI before benchmarking.

---

## F-015 — WITHDRAWN — "SNAPI's 4p3/2 rates violate the Milne relation"

    Original claim: the chromospheric Ca II overionization was caused by a defect in the
    SNAPI reference data (its stored 4p3/2 bound-free rates being 9-22x larger than the
    Milne relation permits given its own byte-identical cross-section table).

**This was a misattribution and is withdrawn.** An independent Lightweaver run on the same
atmosphere agreed with SNAPI's populations (median 3-10%) and disagreed with ours (18-21%),
showing the dominant error was **ours**, not the reference data's. The cause was F-017,
double-counted active-atom bound-free opacity.

The underlying observation about SNAPI's *stored rate matrix* still stands as an observation
— with an identical cross-section table for levels 3 and 4 the ratio R_ki(4)/R_ki(3) should
be ~1.90 and the stored values give 17-40 — but it is **not** the cause of the population
discrepancy, and may well reflect a convention in how SNAPI serialises that matrix rather
than the rates it actually used. It is left in `OPEN_QUESTIONS.md` Q-12 as a curiosity, not
a defect.

**Process lesson, recorded deliberately:** the error was attributing a discrepancy to the
reference data on the strength of an internal-consistency argument, without first obtaining
an independent third implementation. Lightweaver was installed in the project environment
the whole time and would have settled it immediately.
Status: **WITHDRAWN**.

---

## F-006 — HIGH — Saha ratios for recombination frozen at n_e,bg while n_e is iterated

    Class:      B
    Location:   main.py:82-86 (computed once, outside the Λ loop); atoms.py:649 (n_e updated)
    Standard:   HM2014 eq. 9.10 — (n_i/n_k)* ∝ n_e exactly

**Correct:** the recombination rate carries (n_i/n_k)* evaluated at the *current* electron
density, consistently with the collisional rates in the same matrix.

**Implemented:** `lte_ratios_photoionization` is evaluated once from `lte_populations`, which
were computed at `ne_bg`. `solve_SEE` then updates `atmosphere.ne` every iteration — measured
up to a **93% change in a single call** — while the frozen ratio keeps multiplying R_ki. The
collisional rates in the same matrix use the current n_e. The converged state is therefore
not the solution of the coupled problem. The same defect applies to `bg_species` (frozen H⁻,
H I, H II densities feeding the background opacities).

**Impact:** R_ki is wrong by exactly the factor n_e/n_e,bg wherever the electron density
moves. Since (n_i/n_k)* ∝ n_e exactly, the correction is a pure rescale.

**Fix:** rescale `lte_ratios_photoionization` by `atmosphere.ne[k]/atmosphere.ne_bg[k]` at
the point of use — one line, no new state.

**Verified:** `test_saha_ratio_scales_with_ne`; and Test C now leaves n_e unchanged to 1e-9
(it moved by 93% before F-001+F-006).
Status: **PATCHED**.

---

## F-005 — MEDIUM — Hydrogen collision tables flat-extrapolated above 30000 K at 14/82 depths

    Class:      B
    Location:   atoms.py:705 (np.interp clamps at table edges)
    Standard:   RH collision.c interpolates linearly in T and warns outside the table

**Implemented:** `np.interp(Tk, coll.temperatures, coll.rates)` clamps flat beyond the table.
The H CE/CI tables span **3000–30000 K**; the model reaches **1e5 K**. 14 of 82 depths (the
transition region) therefore run on a flat extrapolation of the highest tabulated rate.
Ca II tables span 3000–1e5 K and are fully covered ✔.

**Impact:** hydrogen collisional rates in the transition region are wrong by an unquantified
amount (the true Ω/CE rates continue to vary above 30000 K). Confined to the top 17% of the
model, where H is strongly ionized anyway. Does **not** affect the Ca II benchmark.

**Fix:** none applied — extrapolating tabulated collision data is not defensible without a
physical model (van Regemorter or a wider table), and inventing one would be exactly the
kind of fudge this audit is meant to catch. The remedy is better atomic data.
Status: **RECORDED** (no code change).

---

## F-007 — MEDIUM — Background scattering emissivity lags one Λ-iteration; no acceleration

    Class:      B (convergence rate, not fixed point)
    Location:   formal_solver.py:455; main.py:312
    Standard:   RH includes background scattering in the ALI operator; Ng acceleration
                (Auer 1987) is standard in RH/MULTI/Lightweaver

**Correct in form:** η = κB + σJ, **not** (κ+σ)B — verified at formal_solver.py:453-456. The
continuum source function is right and the scattering *is* inside the iteration, not frozen.

**Implemented:** `atmosphere.J_nu` is assigned only *after* the ray loop (main.py:312), so the
scattering emissivity uses the previous iteration's J. This is plain Λ-iteration on the
scattering term: it converges to the correct fixed point but slowly wherever σ/(κ+σ) → 1 —
the near-UV/violet, which is exactly where Ca II H & K form. Scattering is not included in
the MALI Λ* operator. No Ng acceleration exists.

**Impact:** convergence rate only. Combined with the default `max_tolerance` of 1e-2, a run
can stop while the violet continuum is still drifting.

**Fix:** none applied (performance/robustness, not correctness, per the brief).
Status: **RECORDED**.

---

## F-010 — MEDIUM — Config Γ_rad omits the lower level's radiative width for hydrogen subordinate lines

    Class:      A (in the data, faithfully used by the code)
    Location:   config_H_Ca_Mg_Na.json line broadening.natural; consumed at main.py:262-264
    Standard:   Γ_rad = Σ_{k<j} A_jk + Σ_{k<i} A_ik (HM2014 §8.1)

**Evidence:** cross-check against the Einstein coefficients of the parsed model shows the
config γ equals **Σ_k A_jk exactly** and omits Σ_k A_ik. Ratios γ_cfg/γ_correct:

| line | ratio | | line | ratio |
|---|---|---|---|---|
| H Lyα, Lyβ, 1→4, 1→5 | 0.996–1.001 ✔ | | H Hα 656 | **0.175** |
| Ca II H, K | 0.951–0.955 ✔ | | H Hβ 486 | **0.060** |
| Ca II IR triplet | 0.951–0.955 ✔ | | H Hγ 434 | **0.024** |
| | | | H Paα 1876 | **0.232** |
| | | | H Brα 4052 | **0.276** |

Resonance lines are correct (Σ_k A_ik = 0). **Ca II is correct throughout** — the IR triplet's
lower levels are the metastable 3d states with no permitted downward transition in the model,
so the benchmark is unaffected. Hydrogen subordinate lines are under-specified by 3.6×–42×.

**Impact:** narrower damping wings for the Balmer/Paschen/Brackett lines. Bounded: linear
Stark dominates wherever n_e is appreciable (Hα at the model bottom: Γ_Stark ≈ 2.5e11 s⁻¹
versus a 4.7e8 s⁻¹ error). Only significant in the upper chromosphere/transition region
where Γ_Stark ≈ 5.4e7 s⁻¹ becomes comparable to the missing term.

**Fix:** none — editing atomic data is outside the audit's remit and would decouple the model
from the reference atom files. The remedy, if wanted, is to compute Γ_rad from the model.
Status: **RECORDED**.

---

## F-012 — MEDIUM — CRD used for four transitions the config itself flags as PRD

    Class:      C (legitimate approximation, undocumented departure)
    Location:   Line.type parsed at atoms.py:135 and never read; ψ = φ everywhere
    Standard:   RH/Lightweaver/SNAPI offer angle-averaged PRD (Hummer 1962) or hybrid PRD
                (Leenaarts et al. 2012) for exactly these lines

Affected: **H Lyα, H Lyβ, Ca II H (396.96 nm), Ca II K (393.48 nm)**.

**Expected error:** CRD overestimates the wing source function, filling in the inner wings and
raising the emergent intensity ~0.1–0.5 nm from line centre by tens of per cent, and it cannot
produce the PRD emission-peak asymmetry. **This will dominate the SNAPI comparison for H & K
if SNAPI ran in PRD mode.** The **Ca II infrared triplet is genuinely CRD** in RH/LW/SNAPI too
and is therefore the clean channel for the benchmark.

**Fix:** not implemented (the brief says not to). Documented; first item to reconcile in the
SNAPI setup comparison.
Status: **RECORDED**.

---

## F-013 — MEDIUM — The SNAPI benchmark pipeline cannot run the configuration it claims

    Class:      B (harness)
    Location:   main.py:14; tests/compare_snapi/create_snapi_config.py:76-91

Three independent breakages: (1) `main.py` hardcodes `config_file = 'config_H_Ca_Mg_Na.json'`
and ignores `sys.argv`, so the path passed by `run_full_comparison.py:52` is **silently
discarded** — every previous "benchmark" run used the default 82-point config, not the
128-point FALC one; (2) the generated config omits the `pel` key that `Atmosphere.from_dict`
requires (`KeyError`); (3) it sets `is_active: False` for H, a flag **no code reads** — H is
always solved in NLTE.

Setup mismatches that remain even once it runs: Ca abundance **2.138e-6** (config) versus
**2.1877624e-6** (SNAPI `calcium_ir_rh.cfg`), a 2.3% difference; microturbulence forced to
**0.0** while FALC's depth-dependent v_turb column is available in `falc_model[:,5]`; SNAPI
solved H **and** Ca II in NLTE simultaneously (13 population rows).

**Fix:** `main.py` now accepts an optional config path as `sys.argv[1]` (falling back to the
existing default), `create_snapi_config.py` emits `pel` and the FALC v_turb column, drops the
inert `is_active` flag, and uses the SNAPI abundance. Remaining differences are enumerated in
`audit/BENCHMARK.md`.
Status: **PATCHED** (harness only — no physics change).

---

## F-003 — MEDIUM — Line grids built on a fixed 3 km/s characteristic width under-resolve hot layers

    Class:      B
    Location:   atoms.py:485-486
    Standard:   RH getlambda.c uses the same fixed `VMICRO_CHAR`, but with q_wing chosen
                per line to cover the real profile

`v_micro_char = v_turb` = 3 km/s, while the real Δν_D reaches **13.5×** that at T=1e5 K. For
the lines configured with `q_wing = 30` (H 3→4, 3→5, 4→5) the window spans only **±2.2 real
Doppler widths** at the model top, so the line contributes no opacity beyond that.

**Impact:** after F-002 the residual normalization loss is only 1 − erf(2.21) = 0.18%, so this
is now a *coverage* issue rather than a normalization one. It is the underlying reason F-002
was so violent for those particular lines.

**Fix:** none applied to the grid generator (it matches RH's `VMICRO_CHAR` approach). The
remedy is a configuration change: raise `q_wing` for the infrared hydrogen lines. This is
also the cause of the one J-bar value that will not converge under ray refinement (the
95 nm Lyman line at the 1e5 K top), noted in `test_K_angular_quadrature_converges`.
Status: **RECORDED** (configuration recommendation).

---

## F-008 — LOW — Lower boundary uses I⁺ = B_ν instead of the diffusion approximation

    Class:      C
    Location:   main.py:218, main.py:474
    Standard:   I⁺ = B_ν + μ dB_ν/dτ_ν (RH, Lightweaver, SNAPI)

**Quantified, and the seed lead's premise does not hold for this model.** The emergent-intensity
error is ≈ (μ dB/dτ)·e^{−τ_tot/μ}. Measured total optical depth over the whole 936-point grid
(LTE populations, bottom→top): **minimum τ = 20.2** at 367 nm; τ = 34 at 500 nm, 23 at 400 nm,
4.6e4 at 656 nm. **No point of the spectrum has τ < 5.** With e^{−20} = 2e-9 the boundary
error is **below 1e-8 relative** — there is no measurable μ-dependent bias to find here. A
controlled τ_bot = 8 test gives −0.022% at μ=1 and <1e-5 at μ ≤ 0.5.

**Fix:** implemented anyway (three lines) because it is correct practice and the plain-B
choice is badly wrong for optically thinner models — but it must be understood as robustness,
**not** as a fix for any observed discrepancy.
Status: **PATCHED**.

---

## F-009 — LOW — Code hygiene

    Class:      B/C
    Locations:  see table

| Issue | Location | Disposition |
|---|---|---|
| `eval(configuration["debug"])` vs `configuration.get("debug", False)` — a JSON `"False"` is truthy, so the two gates disagree | main.py:40 vs 534 | **PATCHED** — single helper, both gates agree |
| Config filename hardcoded, `sys.argv` ignored | main.py:14 | **PATCHED** (see F-013) |
| `tau_depth[0,:]` set to χ(0)·Δz under a comment claiming "zero", biasing every saved τ scale | main.py:476 | **PATCHED** — set to 0 |
| `if 'emergent_I_vertical' in locals()` | main.py:366 | **PATCHED** — initialised before the loop |
| Stale docstring: claims scattering is treated as κB, code correctly uses σJ | formal_solver.py:404-406 | **PATCHED** |
| `add_background_opacity_old` — 60 lines of unreachable duplicate physics | formal_solver.py:666-723 | **RECORDED** (left in place; deleting it is a larger diff than the audit warrants) |
| `get_RT_coefficients` depends on `line.vdw_cross` etc. injected as side effects by main.py, so the RT module cannot be exercised standalone — the direct cause of there being no unit tests for it | formal_solver.py:334 ff. | **PATCHED** — extracted to `atoms.init_line_broadening(atoms, atmosphere)`, called from main.py; behaviour identical |
| Voigt/damping block duplicated verbatim between main.py and formal_solver.py | main.py:259-290 vs formal_solver.py:323-355 | **PATCHED** — single `line_profile()` helper used by both |
| `voigt()` is a Python loop over frequencies (~1e4 calls/iteration) | formal_solver.py:273 | **RECORDED** — correctness verified (5e-5 vs `scipy.special.wofz`); vectorising is a performance change, deliberately not bundled |
| Convergence measured only on populations, tolerance 1e-2 | atoms.py:654-675, config | **RECORDED** — no criterion on J̄ and no ‖P·n‖ residual is reported; the 1e-2 default is very loose. Not changed: it alters run behaviour and belongs with the convergence work of F-007 |
| Diagnostic plotting inside the Λ loop dominates runtime | main.py:345-442 | **RECORDED** — gated behind `save_dir`, not the debug flag; not restructured |
| `get_RT_coefficients` is called once per **ray** inside the ray loop, but opacities depend only on depth — a 6× redundant recomputation of the most expensive kernel (the scalar Peach/Fe I loops) | main.py:229,243 | **RECORDED** — caching per depth would cut a Λ-iteration's cost by ~6×; performance only, deliberately not bundled with physics fixes |

---

# Refuted seed leads — investigated, no defect found

These are recorded so they are not re-opened. Each was tested, not merely read.

## R-01 — REFUTED — Voigt truncation is *not* stealing damping-wing weight
Seed lead 1/2. Measured Σ_outside(φw)/Σ_inside(φw) = **1e-9 … 9e-6** for all 15 lines: the
discarded wing is negligible. The truncation matters only because it relocates the pathological
edge weight — the real defect is F-002.

## R-02 — PARTLY REFUTED — the `0.5 * weigths[ir]` *convention* is correct; the *node placement* was not
Seed lead 3. `get_angular_quadrature_1D` returns nodes with **Σw = 2.0** (verified), so
`0.5·w·I` is exactly ½∫₋₁¹I dμ — **there is no factor-of-2 error**, and **Test H**
(isothermal, optically thick, static slab) returns J/B = 1.00000000 at depth and
0.50000000 at the surface, to machine precision.

However, chasing *why* J̄ failed the grid-doubling test uncovered a real defect the
factor-of-2 question would never have found: the nodes were placed across the μ = 0
discontinuity in I(μ). See **F-014**. Test H is blind to this because I(μ) is constant
within each hemisphere in an isothermal slab — a good illustration that passing the
standard normalization test does not certify the quadrature.

## R-03 — REFUTED — Λ* forced to zero at the boundary node is correct and consistent
Seed lead 5. At the starting node the intensity is *prescribed*, so ∂I/∂S_local = 0 exactly —
which is what the solver does there (it never calls `formal_solution` at that node). The
opposite ray direction supplies the non-zero Λ* contribution at that depth. **Test G** confirms
Λ*_kk matches the numerical ∂J_k/∂S_k to **1.1e-10**.

## R-04 — REFUTED — `lte_ratios_photoionization` *is* a pure function of (T, n_e)
Seed lead 7. Proven analytically: the partition functions cancel exactly in n_i*/n_k*, leaving
n_e(g_i/2g_k)(h²/2πm_ek_BT)^{3/2}e^{χ/k_BT} = HM2014 eq. 9.10. It is not contaminated by NLTE
populations, and it is immune to F-001. Computing it outside the loop is legitimate — the
defect is only that it is not rescaled when n_e moves (F-006).

## R-05 — REFUTED — there is no zero frequency in the grid
Seed lead 8. Grid minimum is **7.396e13 Hz**; no zero or negative entry exists.
`hnu_grid[hnu_grid == 0] = 1e-100` (main.py:71) is dead code.

## R-06 — REFUTED — `if ray == np.max(rays)` is diagnostic-only
Seed lead 9. It selects μ = 0.9325 (the largest quadrature node), not μ = 1, but the value is
used **only** for a per-iteration debug plot whose title already says "most vertical ray". The
delivered spectrum comes from the separate genuine μ = 1 pass at main.py:484-498.

## R-07 — REFUTED — the linear-Stark "10⁻⁴ scalar" comment is correct
Seed lead 10. Explicit dimensional analysis: Lightweaver evaluates `… × 1e-4 × n_e[m⁻³]^(2/3)`.
With n_e[m⁻³] = 10⁶·n_e[cm⁻³] and (10⁶)^(2/3) = **10⁴** exactly, the 10⁻⁴ cancels. Dropping it
when passing n_e in cm⁻³ is exactly right.

## R-08 — REFUTED — no catastrophic cancellation in the formal solver
`psi_lin` uses an 8th-order Taylor branch for Δτ ≤ 0.1. Verified over Δτ ∈ [1e-10, 1e2]:
ψ_m, ψ_o correct to full precision, and ψ_m + ψ_o + e^{−Δτ} = 1 to machine precision at every
Δτ. `expm1` is not needed. **Test E** reproduces I(0,μ) = a + bμ to 7.8e-15 (μ=1) … 3.0e-14
(μ=0.1); **Test F** reproduces I = S(1−e^{−τ}) + I₀e^{−τ} to ≤2.9e-14 down to τ = 1e-8.

> **Correction to the audit brief.** The brief states Ψ_O = w₁/Δτ and Ψ_M = w₀ − w₁/Δτ.
> That has M and O interchanged: it gives I_O → S_M as Δτ→∞, which is unphysical. Deriving
> I_O = I_M e^{−Δτ} + ∫₀^{Δτ}S(t)e^{−t}dt with t measured back from O gives
> **Ψ_M = w₁/Δτ, Ψ_O = w₀ − w₁/Δτ**, which is exactly what formal_solver.py:958-959
> implements. The code is right.

## R-09 — REFUTED — the MALI preconditioning is the correct Rybicki–Hummer form
Section 2.12. R_ij = B_ij J̄_eff and R_ji = A_ji(1−Λ*) + B_ji J̄_eff (atoms.py:778-779) is
the analytically-derived form with the S_l dependence moved to the left-hand side — **not**
the "substitute J̄ − Λ*S into the old rates" shortcut. Λ̄* correctly carries the
χ_line/χ_total chain-rule factor, and `S_old` correctly uses the outer-iteration populations.

## R-10 — REFUTED — the SEE matrix assembly, sign convention and conservation row are correct
Section 2.1. The conservation row **is** chosen per depth point (`np.argmax(old_pops_k)`), the
next-ion level **is** included in the sum, and the row is scaled for conditioning. No negative
populations were produced in any test.

## R-11 — REFUTED — no line-of-sight velocity exists to be mishandled
Seed lead 6. The `Atmosphere` dataclass carries **no velocity field at all**, so the absence of
a μ·v_los/c term in `dop_freq` is correct for the static FAL-C models in use. It is a hard
limitation (any model with velocities would need a per-ray profile), documented in 02_physics.md
§2.3 — but it is not a bug in the current configuration.

## R-12 — REFUTED — Einstein relations and the f→A conversion are exact
max |A/((2hν³/c²)B) − 1| = 2.2e-16 and max |g_iB_ij/(g_jB_ji) − 1| = 1.1e-16 over all 15 lines;
absolute values reproduce NIST (A(Lyα)=4.696e8, A(Hα)=4.407e7, A(Ca II K)=1.466e8 s⁻¹).

## R-13 — REFUTED — background scattering *is* treated as scattering
Section 2.10. `ems = kappa*B + sigma*atmosphere.J_nu[iz,:]` — η = κB + σJ, not (κ+σ)B, and it
is inside the iteration. Only the one-iteration lag is imperfect (F-007). The *docstring*
claiming otherwise is stale and was corrected.

## R-14 — REFUTED — all four broadening recipes are dimensionally and numerically correct
Section 2.13. Unsold (including the reduced-mass factors, the He number-ratio term and the
T^0.3 exponent), Barklem/ABO (σ in a₀², v₀ = 10⁶ cm s⁻¹, the factor 2 half-width→full-width,
the T^{0.5(1−α)} exponent), quadratic Stark (11.37 Lindholm coefficient, Traving C₄, T^{1/6},
n_eff, the 28.0 amu average perturber as in RH/LW) and linear Stark all check out, each with
an explicit dimensional analysis reducing Γ to s⁻¹.

## R-15 — PARTLY REFUTED — no negative opacity in LTE, but it *does* occur in NLTE
Measured over all 82 depths × 936 frequencies with LTE populations: **0 negative opacity
samples, 0 negative emissivity samples**. So the LTE claim stands.

**But the converged NLTE run contradicts it**: `main.py` printed "Negative absorption at
depth 127" on essentially every iteration (25 of the first 26), always at the topmost point,
together with hundreds of negative-source-function warnings. `S = emis/abs`
(formal_solver.py:920-921) is unguarded and Δτ < 0 makes `exp(-Δτ) > 1` — an amplifying step
— at the last point of every upward ray. The code only warns; the corrections at
formal_solver.py:390,393 are commented out. Cause and magnitude undetermined: see
`OPEN_QUESTIONS.md` Q-11.
