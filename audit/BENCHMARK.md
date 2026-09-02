# Phase 4 — Benchmark against SNAPI

Working data: `tests/compare_snapi/snapi_firtez_comparison_data.npz`
Comparison driver: `tests/compare_snapi/compare_against_snapi.py`
Config generator: `tests/compare_snapi/create_snapi_config.py` (repaired, see F-013)

---

## 1. What is actually in the benchmark

Read from the files, not assumed:

| Dataset | Shape | Content |
|---|---|---|
| `falc_model` | (128, 12) | FAL-C. Col 0 log τ₅₀₀, 1 z [cm], 2 T [K], 3 p_g, 4 p_el, 5 v_los (**identically zero**), 6 ρ, 7 B, **8 v_turb [cm/s]**, 10–11 constants π/4, π/3. Ordered **top → bottom**. |
| `atm_lvl_pops` | (97, 128, 13) | 97 iterations × 128 depths × (6 H + 6 Ca II + n_e) |
| `debug_J`, `debug_L`, `debug_norm` | (97, 128, 15) | J̄ (**per wavelength**), Λ*, and the profile normalisation, for all 15 (i<j) pairs of the 6-level Ca II atom |
| `rate_matrix` | (97, 128, 6, 6) | rate matrix before the conservation row is substituted |
| `synth_spectrum` | (501, 5) | λ [cm], I, Q, U, V — **covers 853.7–854.7 nm only** |

Two facts that shape everything below:

1. **The synthesised spectrum is the Ca II 854.2 nm infrared line alone** — not H & K. The
   IR triplet is treated in CRD by RH, Lightweaver and SNAPI as well, so **F-012 (this code
   being pure CRD) does not contaminate the spectrum comparison.** It could still influence
   the populations through H & K, which are formed regardless of what was synthesised.
2. **The atmosphere reaches only 9168 K** (photosphere + chromosphere, no transition region).
   So F-005 (H collision tables clamped above 30000 K) and F-003 (line windows
   under-resolved at 1e5 K) are **both irrelevant to this benchmark**.

---

## 2. Are the two codes being asked the same question?

This is where most "benchmark disagreements" actually live, so it was checked item by item
against the SNAPI atom file (`calcium_ir_rh.cfg`) and the FAL-C model, not assumed.

### Verified identical

| Quantity | Result |
|---|---|
| **Ca II level energies** | Exact. SNAPI stores eV; 1.692408, 1.699932, 3.123349, 3.150984 eV convert precisely to our 13650.19, 13710.88, 25191.51, 25414.40 cm⁻¹ |
| **Statistical weights g** | Exact: 2, 4, 6, 2, 4 (+1 for Ca III) |
| **J, L** | Exact (SNAPI stores 2J: 1, 3, 5, 1, 3 vs our 0.5, 1.5, 2.5, 0.5, 1.5) |
| **Ca abundance** | Exact after F-013: 2.1877624e-06 (was 2.138e-06, 2.3% off) |
| **Bound-free cross-sections** | **Exact — ratio 1.000 at threshold for all five levels** (SNAPI tabulates λ in Å, we in nm; σ agrees to 4 significant digits, e.g. 2.0363e-19 cm² for the ground state, 6.1484e-18 for 3d) |
| **Collisional rates** | **All 20 sampled (i→j, T) pairs agree to 0.03%.** SNAPI tabulates the downward rate C_ji/n_e directly; our Ω converted as C₀Ω/(g_j√T) with C₀ = 8.6291e-6 reproduces it. The uniform 1.0003 factor is the 4-digit rounding of Ω in our JSON |
| **Electron density** | **Median ratio 1.0000** (min 0.960, max 1.002) between `compute_background_eos` and SNAPI's converged n_e — see below |
| **Microturbulence** | Now matched: FAL-C column 8, 1.79 → 6.75 km/s (F-013 fixed reading the wrong column) |
| **Depth grid** | Identical 128 points, same z values |
| **Active atoms** | Both solve **H and Ca II in NLTE simultaneously** (SNAPI's 13 population rows confirm it) |

### The electron-density result (resolves what was the largest setup risk)

| z (km) | T (K) | n_e (FAL-C p_el) | n_e (SNAPI) | n_e (our EOS) | ours/FAL-C | SNAPI/FAL-C |
|---|---|---|---|---|---|---|
| −90 | 9168 | 3.044e15 | 3.086e15 | 3.086e15 | 1.014 | 1.014 |
| 230 | 4922 | 3.018e12 | 2.613e12 | 2.515e12 | 0.833 | 0.866 |
| 550 | 4517 | 1.868e11 | 1.607e11 | 1.545e11 | 0.827 | 0.860 |
| 934 | 5859 | 1.273e11 | 3.406e11 | 3.408e11 | 2.676 | 2.675 |
| 1350 | 6766 | 1.122e11 | 5.113e11 | 5.113e11 | 4.556 | 4.557 |
| 1942 | 8164 | 7.669e10 | 1.820e11 | 1.820e11 | 2.373 | 2.373 |

**Both codes depart from the FAL-C `p_el` column by the same large factor** (median 2.48, up
to 4.69 in the chromosphere), and agree with each other to 0.02% in the median. So this
code's practice of parsing `pel` and then ignoring it, solving n_e from a 92-element LTE
equilibrium on (T, p_g), **matches SNAPI's convention**, and `compute_background_eos` is
validated against an independent implementation.

### Remaining differences — explicitly listed

| Difference | Status |
|---|---|
| **Oscillator strengths, Γ_rad, broadening recipes, line wavelength quadratures** | **Cannot be verified.** SNAPI's atom file contains only levels, bound-free tables and collisional rates — no `F=`, `GF`, `AJI`, `GRAD`, `NLAMBDA`, `QCORE` or `QWING` entries anywhere. SNAPI must supply these internally or from another file that was not provided. This is the largest unresolved setup uncertainty. |
| **CRD vs PRD** | Unknown for SNAPI (Q-01). Immaterial for the 854.2 nm spectrum; possibly material for the populations via H & K. |
| **Angular quadrature** | Not matched. Ours raised to `n_gaus = 12` per F-014; SNAPI's setting unknown (Q-04). |
| **Ca III ionization energy** | 11.868 eV (SNAPI `EION`) vs 11.8757 eV implied by our Ca III level at 95785.47 cm⁻¹ — a 0.065% difference. |
| **Background opacity** | Ours is the Wittmann/Mihalas suite. SNAPI's is unknown. Affects the continuum against which the line is measured. |
| **Ca I stage** | Absent from both models (all Ca distributed over Ca II + Ca III). |

---

## 3. Result of the comparison run

*(Filled in from the run of `main.py config_compare_snapi.json` after the F-001, F-002,
F-006, F-008, F-013 and F-014 fixes.)*

Run: `main.py config_compare_snapi.json`, 128-point FAL-C, H and Ca II both NLTE,
`n_gaus = 12`, 100 iterations. It reached max |Δn/n| = **4.3e-4** and stopped on the
iteration limit, not the 1e-4 tolerance — still falling ~6% per iteration. Convergence is
slow (F-007: the scattering emissivity lags one iteration, and there is no Ng acceleration).
The residual is far below the differences discussed here, so it does not affect the
conclusions.

### 3.1 Emergent spectrum — the air/vacuum correction

**SNAPI tabulates AIR wavelengths.** Its line core sits at exactly 854.2090 nm, the textbook
air value; Edlén (1966) converts that to 854.4437 nm, against this code's
λ₀ = hc/ΔE = **854.4438 nm** — agreement to **0.0001 nm**. Comparing without converting
offsets the two spectra by 0.26 nm, which is wider than the line core, and measures core
against wing:

| metric | uncorrected | **corrected** |
|---|---|---|
| core RMS (\|Δλ\| < 0.03 nm) | 52.20% | **2.29%** |
| wing RMS (\|Δλ\| > 0.03 nm) | 38.11% | **1.66%** |
| overall RMS | 45.33% | **1.97%** |
| line-core intensity | −23.14% | **−2.38%** |
| core wavelength | 854.4438 vs 854.1818 | **exact** |

This is the textbook case the brief warns about — a setup mismatch that would have sent the
whole diagnosis chasing ghosts. **The emergent Ca II 854.2 nm profile agrees with SNAPI to
2.0% RMS**, with the core 2.4% too deep.

### 3.2 Populations — excellent in the photosphere, diverging with height

| z (km) | (ours − SNAPI)/SNAPI for Ca II ground |
|---|---|
| −90 | +0.74% |
| 6 | +0.00% |
| 198 | −0.13% |
| 406 | −0.08% |
| 598 | +0.15% |
| 806 | −8.10% |
| 998 | −22.52% |
| 1206 | −42.70% |
| 1494 | −62.43% |
| 1942 | −67.79% |

Total Ca is conserved identically in both codes (median ratio 1.0001, range 0.998–1.009), so
this is **not** a normalisation problem: it is an ionization-balance difference. The Ca II
fraction bottoms out at **0.184 in this code versus 0.588 in SNAPI** — we overionize Ca II
to Ca III by a factor ≈7.6 in the ratio at the top of the chromosphere. Level 5 (Ca III)
therefore shows a 167% median difference while levels 0–4 show 10–16%.

J̄ per transition (SNAPI's `debug_J` is per wavelength; converted with J_ν = J_λ λ²/c):

| line | mean \|rel\| | median | max |
|---|---|---|---|
| Ca II H 396.96 nm | 17.6% | 16.0% | 40.8% |
| Ca II K 393.48 nm | 16.0% | 11.4% | 41.7% |
| Ca II 866.45 nm | 11.1% | 8.1% | 30.8% |
| Ca II 850.04 nm | 7.7% | 5.2% | 23.0% |
| Ca II 854.44 nm | 10.9% | 7.1% | 33.3% |

### 3.3 Localising the residual — and what it is not

Following the brief's rule that a residual must be interpreted physically, the
ionization/recombination channel was compared against SNAPI's `rate_matrix` directly. Its
convention was first established rather than assumed: **SNAPI's M[j,i] is the rate i→j**, and
in the deep layers M[i,5]/M[5,i] reproduces *our* Saha–Boltzmann ratio to 1% for levels
0, 1, 3 and 4 — so the convention and the Saha factor both match.

| z (km) | level | R_ik ours/SNAPI | R_ki ours/SNAPI |
|---|---|---|---|
| 406 | 0 | 0.900 | 0.902 |
| 406 | 1 | 0.850 | 0.971 |
| 406 | 4 | 0.030 | **0.070** |
| 806 | 0 | 0.521 | 0.976 |
| 806 | 4 | 0.149 | **0.081** |
| 1206 | 0 | 0.982 | 0.926 |
| 1206 | 4 | 0.815 | **0.084** |
| 1942 | 0 | 0.454 | 0.759 |
| 1942 | 4 | 1.347 | **0.078** |

**Recombination to the Ca II 4p levels is a factor ≈12 smaller than SNAPI's at every
depth**, including 406 km where every other quantity agrees to 1%. Too little recombination
is exactly the right sign to produce the observed overionization.

**Causes eliminated by direct test** (each was checked, not assumed):

1. **The Saha–Boltzmann factor.** (n₄/n₅)* from `lte_populations` equals the analytic
   n_e(g_i/2g_k)(h²/2πm_ek_BT)^{3/2}e^{χ/k_BT} to **1.000000**, and the ratio between levels
   0 and 4 is the Boltzmann factor 139.46 exactly, as it must be.
2. **The bound-free cross-sections.** Ours and SNAPI's agree to **1.000** over 35–142 nm
   (they differ only below ~30 nm, where our table simply starts later and the contribution
   is negligible).
3. **The electron density.** Converged n_e/n_e,bg = 0.77–1.07 (median 0.9915), and
   n_e/n_e,SNAPI median 0.968 — so the F-006 rescale is ≈1 and cannot explain a factor 12.
   *(An earlier version of this diagnostic used n_e,bg and would have misattributed the
   result; `main.py` now saves the converged n_e precisely so this cannot recur.)*
4. **Frequency-grid resolution near the edge.** The obvious suspect: levels 3 and 4 have only
   4–5 grid points within k_BT/h of their 141.7/142.1 nm thresholds (spacing
   142.10 → 141.66 → 140 → 135 nm). But against dense `scipy.quad` the grid **overestimates**
   the spontaneous recombination integral by a factor **1.41** — the wrong direction, and an
   order of magnitude too small. **This hypothesis does not survive**, and is recorded as
   refuted rather than offered as the explanation.
5. **My own arithmetic.** R_ki was recomputed from first principles at z = 1206 km:
   (n₄/n₅)* = 4.451e-3, the spontaneous integral 8.748 (dense-quad reference 6.220), the
   stimulated term 1.29e-8, giving R_ki = 3.894e-2 against SNAPI's 4.640e-1.

**What this implies.** For SNAPI to obtain its value with the same cross-sections and the
same Saha factor, its spontaneous recombination integral would have to be ≈16× the exact
value computed from those shared cross-sections. That is not achievable with the same
formula, so the difference is **definitional or physical, not a numerical error in this
code's evaluation of the equation it implements**.

Two hypotheses remain, and I can distinguish neither from the material provided — they are
recorded as hypotheses, not conclusions:
- SNAPI includes a recombination channel absent from this model entirely, most plausibly
  **dielectronic recombination** to Ca II 4p, which is a real process and can contribute an
  order of magnitude at chromospheric temperatures.
- SNAPI's `rate_matrix` uses a different normalisation for the bound-free entries than for
  the bound-bound ones (the deep-layer detailed-balance check would not distinguish this if
  the same convention is applied to both directions).

Resolving it requires SNAPI's source or documentation for how `rate_matrix` bound-free
entries are formed. This is `OPEN_QUESTIONS.md` Q-12 and is the single most valuable
remaining question.

### 3.4 Summary of metrics

| Quantity | Result |
|---|---|
| LTE continuum / EOS (n_e) | **0.02% median**, worst 4% |
| Bound-free cross-sections | **exact** (1.000) |
| Collisional rates | **0.03%** |
| Ca II populations, z ≤ 600 km | **≤ 0.15%** |
| Ca II populations, z ≥ 1200 km | 43–68% (overionization) |
| J̄, Ca II IR triplet | 5–8% median |
| J̄, Ca II H & K | 11–16% median |
| **Emergent 854.2 nm, core RMS** | **2.29%** |
| **Emergent 854.2 nm, wing RMS** | **1.66%** |
| **Line-core intensity** | **−2.38%** |
| Line-core wavelength | exact |


---

## 4. Honest statement of what this benchmark can and cannot establish

**Can:** that the equation of state, the atomic data as parsed, the bound-free physics and
the collisional physics agree with an independent code to a fraction of a per cent. That is
a real and non-trivial verification, and it was only possible after F-013 made the harness
run at all.

**Cannot:** attribute any residual population or line-profile difference to a specific piece
of physics, because the bound-bound atomic data (f-values, Γ_rad, damping) and the
redistribution treatment could not be read from the material provided. A residual in the
854.2 nm line core could equally be a J̄ difference, an f-value difference, or a damping
difference, and this dataset cannot separate them.

The single most valuable addition would be **SNAPI's bound-bound transition data and its run
settings** (redistribution, ray count). With those, the remaining comparison becomes
diagnostic rather than merely descriptive.

### 3.5 Resolving the residual — three causes, and both problems trace to Ly-alpha

Two setup facts supplied by the author were tested directly: **SNAPI fixes the electron
density to its initial values**, and **SNAPI used 3 up + 3 down rays**. A `fix_electron_density`
option was added (`config_snapi_match.json`, `n_gaus: 6`) and the run repeated.

Independent confirmation of the first: with the option enabled our n_e equals `ne_bg` exactly
(1.000000), and that value matches SNAPI's converged n_e to a **median of 1.0000**. So SNAPI's
electrons are indeed the initial EOS values, and our EOS reproduces them.

**Cause 1 — SNAPI's 4p3/2 bound-free rates violate the Milne relation (F-015).** Its own atom
file gives levels 3 and 4 a byte-identical cross-section table, which forces
R_ki(4)/R_ki(3) ≈ 1.90. Ours obeys it (2.4-2.9); SNAPI's does not (17-40). Our 4p1/2 rates
agree with SNAPI to 3-8%; only the 4p3/2 channel disagrees, and it carries 56% of SNAPI's
total recombination at 1206 km. Re-solving SNAPI's own SEE with that channel restored to Milne
consistency reduces the median population discrepancy by **29%** (0.233 → 0.166).

**Cause 2 — the electron-density convention.** Freezing n_e as SNAPI does raises the Ca II
fraction at the top from 0.184 to 0.257 and reduces the ground-state difference at 1942 km
from −67.8% to −59.5%. Real, and in the expected direction (recombination scales with n_e,
and our evolving n_e fell to 0.77 × ne_bg at the top), but secondary.

**Cause 3 — WITHDRAWN.** An earlier revision of this section attributed the dominant
remainder to CRD Ly-alpha over-illuminating the Ca II 3d photoionization edges (which do lie
inside the Ly-alpha window, at 121.750/121.840 nm). The author confirmed **SNAPI did not run
PRD**, and Lightweaver's H lines are all `VoigtLine` (CRD) as well, so a CRD-vs-PRD mismatch
cannot be the explanation. The wavelength overlap is real but is not the cause.

**Cause 3 (actual) — F-017: the bound-free opacity of ACTIVE atoms was counted twice.**
`get_RT_coefficients` adds each active atom's continua, and `add_background_opacity` then
added the same transitions again (`_opac_h_hydrogenic` for H, `_opac_metals_luke` for Ca II).
Measured, the duplicate equals the original to within 3% across the Lyman continuum, i.e. the
opacity there was **nearly a factor 2 too large**. Excess Lyman-continuum opacity thermalises
the far-UV field to the local, chromospherically hotter Planck function, so J runs high and
the error grows with height: H ground-state photoionisation was 1.05x Lightweaver's at
406 km rising to **9.57x at 1494 km**, while levels 1 and 2 (edges longward of the Lyman
continuum) stayed flat at 0.80 and 0.99. RH and Lightweaver both exclude active atoms from
the background for exactly this reason. See `FINDINGS.md` F-017.

**Cause 4 — hydrogen was held in LTE by SNAPI, and this code had no way to do that.**
Verified directly from the benchmark: SNAPI's H departure coefficients are 0.968-1.009 and
its H populations are **bit-identical between iteration 1 and iteration 97** (max relative
change exactly 0.000e+00) while Ca II changes by 264x. An `is_active` flag existed in the
configuration but was read by nothing; active/passive support was implemented (F-013/F-017
commit). The H treatment is a large lever: Lightweaver's own Ca II fraction minimum moves
from 0.638 (H active) to 0.389 (H detailed-static) on the same atmosphere.

### 3.6 Final state of the comparison

Two independent references were used, because the first attempt misattributed a discrepancy
to the benchmark data on an internal-consistency argument alone (see F-015, withdrawn).
Lightweaver 0.16.1 was run on the identical stratification with the identical RH `H_6` and
`CaII` atoms — f-values, quadratures and collision data verified identical, all lines
`VoigtLine`. Its detailed-static hydrogen is confirmed LTE to 0.069% against SNAPI's, so it
is the correct analogue of SNAPI's setup.

Ca II levels 0-4, median (mean) |X/reference − 1| over all 128 depths:

| configuration | vs SNAPI | vs Lightweaver (matched) |
|---|---|---|
| ours, at the start of this benchmark | 13.72% (30.60%) | 19.32% (28.05%) |
| ours, after F-017, H passive, vmc 6.75 km/s | 6.14% (10.89%) | 3.47% (5.55%) |
| **ours, after F-017 + F-018, H passive** | **4.93% (10.32%)** | 5.98% (7.18%) |
| Lightweaver, H detailed-static | 10.18% (13.68%) | — |
| Lightweaver, H active | 6.06% (7.24%) | — |

Emergent Ca II 854.2 nm (once the air/vacuum scale is corrected): core RMS 2.29%, wing RMS
1.66%, line-core intensity −2.38%, core wavelength exact.

**The honest bottom line.** The two reference codes disagree with each other by **10.18%
median** on this problem, so our ~5% sits inside their mutual spread. Further tuning against
either one is not meaningful without knowing why they differ, which would need SNAPI's
background-opacity treatment and its hydrogen handling in more detail than the benchmark
files provide. What can be stated is that after F-017 our transfer reproduces Lightweaver's
J-bar to ratio 1.000 given the same populations, and Lightweaver's converged solution is now
a fixed point of our iteration.

### 3.7 Note on the negative opacity

With hydrogen passive at LTE the negative-opacity warnings disappear entirely (0 per run,
against 99 before), because LTE hydrogen cannot invert. This independently confirms the
F-016 diagnosis that the negative total opacity originated in the n=4/n=5 population
inversion of our NLTE hydrogen solution at the topmost grid point. F-016's underlying
concern — that one grid cell spans dtau = 470 where the Ly-alpha thermalization layer is
tau ~ 688 deep — still stands for any run with hydrogen active.

