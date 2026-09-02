# Phase 2 — Equation-by-equation audit

Format per item: (i) correct equation, (ii) code as implemented, (iii) difference,
(iv) failure class A/B/C, (v) severity. Findings are cross-referenced to `FINDINGS.md`.

Verification standard: Hubeny & Mihalas 2014 (HM2014), Mihalas 1978, Rybicki & Hummer
(1991, 1992), Uitenbroek 2001 (RH), Osborne & Milic 2021 (Lightweaver, "LW").

---

## 2.1 Statistical equilibrium — **CORRECT**

Correct (HM2014 eq. 14.6):
$$\sum_{j\neq i} n_j P_{ji} - n_i \sum_{j\neq i} P_{ij} = 0,\qquad P_{ij}=R_{ij}+C_{ij}$$

Implemented, `atoms.py:787-792`:
```
total_departure_rate_from_i = np.sum(R_matrix + C_matrix, axis=1)
A_matrix[i, j] = R_matrix[j, i] + C_matrix[j, i]
A_matrix[i, i] -= total_departure_rate_from_i[i]
```
The diagonals of `R_matrix`/`C_matrix` are never written, so `total_departure_rate_from_i[i]`
= Σ_{j≠i} P_ij and `A[i,i]` = −Σ_{j≠i} P_ij. Row *i* reads Σ_j n_j P_ji − n_i Σ_j P_ij = 0.
**Matches exactly, sign convention correct.**

- Conservation row: replaced at `max_pop_idx = np.argmax(old_pops_k)` — **chosen per depth
  point** (`atoms.py:795`), as required for conditioning. ✔
- Row scaled by `total_departure_rate_from_i[max_pop_idx]` (`atoms.py:799-803`) — a
  reasonable row-equilibration; no column scaling, but the matrix conditioning was adequate
  in every case tested. ✔
- The continuum / next-ion level *is* included in the conservation sum (it is an ordinary
  entry of `atom.levels`). ✔
- Negative populations are clamped to `1e-100` (`atoms.py:815`). Did **not** fire in any test
  performed. Retained as a watch item, not a finding.

**Class: none. Verified by Test C (detailed balance) to 1e-5 (H) / 5e-8 (Ca II) in shape.**

---

## 2.2 Bound–bound radiative rates — **CORRECT**

$$A_{ji}=\frac{2h\nu^3}{c^2}B_{ji},\qquad g_iB_{ij}=g_jB_{ji},\qquad
A_{ji}=\frac{8\pi^2e^2\nu^2}{m_ec^3}\frac{g_i}{g_j}f_{ij}$$

Implemented `atoms.py:372-375`. **Verified to machine precision for all 15 parsed lines**
(max |A/((2hν³/c²)B)−1| = 2.2e-16; max |g_iB_ij/(g_jB_ji)−1| = 1.1e-16), and the absolute
values reproduce NIST: A(Lyα)=4.696e8, A(Hα)=4.407e7, A(Ca II K)=1.466e8,
A(Ca II 854.2)=9.92e6 s⁻¹.

**Class: none.**

---

## 2.3 Line profile and normalization — **CRITICAL BUG (F-002)**, plus F-003

$$\phi(\nu)=\frac{H(a,v)}{\sqrt{\pi}\,\Delta\nu_D},\quad
\Delta\nu_D=\frac{\nu_0}{c}\sqrt{\frac{2k_BT}{m}+\xi^2},\quad \int\phi\,d\nu=1$$

**Sub-items verified correct:**
- `voigt()` (`formal_solver.py:262-303`) returns H(a,v)=Re[w(v+ia)]: max relative error
  5e-5 against `scipy.special.wofz` for a ∈ [1e-4, 1]. Gaussian limit H(0,2)=e⁻⁴ exact;
  Lorentz wing H(0.1,50)=2.258e-5 vs a/(√π v²)=2.257e-5. ✔
- `Δν_D` uses **2**k_BT/m (most-probable speed) ✔ and ξ as a velocity, not a FWHM ✔
  (`atoms.py:408-409`).
- `a = Γ/(4πΔν_D)` ✔ — Γ is a full damping rate in s⁻¹, Lorentzian HWHM in ν is Γ/4π.
- The analytic profile **is** correctly normalized: ∫φdν over the truncation window computed
  with `scipy.integrate.quad` gives 0.999989–0.999999 for every line/depth tested.

**The truncation is NOT the problem.** Measured Σ_outside(φw)/Σ_inside(φw) = 1e-9 to 9e-6 for
all 15 lines: the discarded damping-wing weight is negligible, contrary to the seed lead.

**The problem is the quadrature weights (F-002, class B, CRITICAL).**
`create_frequency_grid` builds *global* trapezoid weights over the union grid
(`atoms.py:538-541`), and these are then used for a *per-line* integral
(`main.py:290,303,305`; `formal_solver.py:355`). The outermost point of each line's window
gets `w = 0.5(ν[i+1] − ν[i−1])`, where ν[i+1] is the next point of the **global** grid — i.e.
across the empty gap separating this line's frequency block from the next block in the
spectrum. Measured for H Brα (4052 nm) at iz=81 (T=1e5 K):

| global idx | v = (ν−ν₀)/Δν_D | local spacing | trapezoid weight | φ | contribution |
|---|---|---|---|---|---|
| 18 (window edge) | 2.210 | 4.0e9 Hz | **2.88e13 Hz** | 4.25e-13 | **12.217** |
| 16 | 0.476 | 4.0e9 Hz | 4.02e9 Hz | 4.47e-11 | 0.180 |
| … | | | | | |
| **Σ** | | | | | **13.306** |

One point — the window edge — carries **92% of the normalization integral**, because its
weight is 7000× the local grid spacing. ∫φdν per line, natural damping, on the global grid:

| line | λ₀ (nm) | iz=0 | iz=41 | iz=81 |
|---|---|---|---|---|
| H 1-2 Lyα | 121.57 | 1.0020 | 1.0019 | 1.0031 |
| H 1-5 | 94.98 | 1.1020 | 1.1060 | 1.0781 |
| H 2-5 | 434.17 | 1.0133 | 1.0131 | 1.0162 |
| H 3-4 Paα | 1875.63 | 1.0053 | 1.0052 | **1.7505** |
| H 3-5 | 1282.17 | 1.0612 | 1.0617 | **1.7813** |
| **H 4-5 Brα** | **4052.29** | **1.1168** | **1.1121** | **13.3063** |
| Ca II H | 396.96 | 1.0140 | 1.0143 | 1.0117 |
| Ca II K | 393.48 | 1.0140 | 1.0143 | 1.0117 |
| Ca II 866.5 | 866.45 | 1.0226 | 1.0225 | 1.0249 |
| Ca II 854.4 | 854.44 | 1.0043 | 1.0042 | 1.0052 |

Consequences of dividing by this number (`voigt_norm = voigt_line/Σ(voigt_line·w)`):
1. Line opacity χ_line and emissivity η_line are **deflated by up to a factor 13**.
2. J̄ = Σ w φ I / Σ w φ becomes a weighted average of I dominated by the window-edge point,
   i.e. **J̄ for the isolated IR hydrogen lines is essentially the local continuum intensity**,
   not the line-core radiation field. The bound-bound rates for those transitions are garbage.
3. Λ̄* is deflated by the same factor, degrading the MALI preconditioning.
4. Ca II lines — the benchmark target — are deflated by a uniform 1.2–2.5%.

This is the answer to the author's TODO ("why does this matter so much"): the sensitivity to
the truncation mask is not about wing physics, it is that the mask *relocates the enormous
edge weight*, and moving that one point moves the normalization by tens of percent.

RH (`getlambda.c`, `wlamb`) and Lightweaver (`Transition.compute_phi`, `wphi`) both integrate
each transition over **its own** wavelength subgrid with weights confined to that subgrid,
which is why neither exhibits this failure. Numerical renormalization itself is standard
practice (LW's `wphi`) and is retained.

**Fix (F-002):** compute per-line trapezoid weights restricted to the line's own window, with
half-intervals at the two ends; use them for the profile normalization, J̄ and Λ̄*.

**F-003 (class B, MEDIUM):** `create_frequency_grid` builds the line grid on a *fixed*
characteristic Doppler width `v_micro_char = v_turb` = 3 km/s (`atoms.py:485-486`), matching
RH's `VMICRO_CHAR`. But the real Δν_D reaches 13.5× that at T=1e5 K, so for the lines
configured with `q_wing = 30` (H 3-4, 3-5, 4-5) the window spans only ±2.2 *real* Doppler
widths at the top of the atmosphere. After F-002 the residual normalization loss is only
1−erf(2.21) = 0.18%, but the line contributes no opacity at all beyond ±2.2 Doppler widths
there. Recommendation: raise `q_wing` for the infrared hydrogen lines; add a diagnostic.

**F-004 (class C, documented):** the profile has no line-of-sight velocity term and no μ
dependence (`main.py:260`, `formal_solver.py:324`). **Confirmed:** the `Atmosphere` dataclass
carries no velocity field at all, so for the static FAL-C models in use this is *correct*, not
a bug. It is a hard limitation: any model with velocities would need
v = (ν − ν₀ − ν₀ μ·v_los/c)/Δν_D and a per-ray profile.

---

## 2.4 Line opacity and emissivity — **CORRECT**

$$\chi^{\rm line}_\nu=\frac{h\nu_0}{4\pi}(n_iB_{ij}-n_jB_{ji})\phi_\nu,\qquad
\eta^{\rm line}_\nu=\frac{h\nu_0}{4\pi}n_jA_{ji}\phi_\nu$$

`formal_solver.py:361-362`, `main.py:296`. Stimulated emission is correctly treated as
negative absorption ✔. ψ = φ (CRD) ✔.

**Consistency between `get_RT_coefficients` and the J̄ loop:** the damping sum, `a_damp`,
Voigt evaluation, truncation mask and renormalization are duplicated *verbatim*
(`main.py:259-290` vs `formal_solver.py:323-355`) and are evaluated at the same `atmosphere.ne`
and the same `atom.populations` within a ray sweep. They are therefore identical, and the MALI
preconditioning is self-consistent. ✔ (Duplication is a maintenance hazard, not a bug —
noted in F-009.)

**Negative total opacity:** measured over all 82 depths × 936 frequencies with LTE
populations — **0 negative opacity samples, 0 negative emissivity samples**. If it did occur,
`formal_solution` would divide by a near-zero `abs_O` (`formal_solver.py:920-921`, unguarded)
and produce Inf/NaN; the warning at `formal_solver.py:388-393` prints but does not correct.

**Class: none for the equations.** The `h*line.nu0` vs `h*ν` choice (TODO at
`formal_solver.py:360`) is the standard approximation used by RH/LW; error ~Δν_D/ν₀ ≲ 1e-4.

---

## 2.5 Bound–free rates — **CORRECT equations, one consistency defect (F-006)**

$$R_{ik}=4\pi\!\!\int_{\nu_0}^\infty\!\!\frac{\alpha_\nu}{h\nu}J_\nu d\nu,\quad
R_{ki}=4\pi\Big(\frac{n_i}{n_k}\Big)^{\!*}\!\!\int_{\nu_0}^\infty\!\!\frac{\alpha_\nu}{h\nu}
\Big(\frac{2h\nu^3}{c^2}+J_\nu\Big)e^{-h\nu/k_BT}d\nu$$

`main.py:322-334`. **Matches exactly** (`hnu3_grid = hν³`, so `2*hnu3/c²` = 2hν³/c²). ✔

**The Saha–Boltzmann ratio is genuinely a pure function of (T, n_e)** — proven analytically:
writing n_i and n_k from `compute_lte_populations`, the partition functions **cancel exactly**
in the ratio, leaving
$(n_i/n_k)^* = n_e (g_i/2g_k)(h^2/2\pi m_ek_BT)^{3/2}e^{\chi/k_BT}$, i.e. HM2014 eq. 9.10.
It is therefore *not* contaminated by NLTE populations ✔ and is *not* affected by F-001 ✔.
`g_k` is the statistical weight of the next ion's ground state (H II g=1, Ca III g=1) ✔,
and the factor 2 for free-electron spin is present in `Phi = (2/ne)·…` ✔.

- α_ν ≡ 0 below threshold: ✔ (`atoms.py:220,259`, λ > λ_edge ⇒ 0).
- Gaunt factor: Seaton (1960) as in RH ✔ (`atoms.py:262-288`).
- Interpolation of tabulated α: `np.interp` with `left=0.0, right=0.0` plus explicit zeroing
  outside [λ_min, λ_edge] and clamping of negative interpolants ✔ — no spurious extrapolation.
- α ≡ 0 below `minWavelength` truncates the cross-section at high frequency (22.8 nm for the
  Lyman continuum). Negligible in a solar atmosphere; **class C, LOW**.

**Continuum quadrature weights:** the bf integral reuses the same global trapezoid weights
(`main.py:330-331`). Unlike the line case this is *defensible* — it is a genuine trapezoid rule
on the union grid, and α is evaluated at every union point. But the union grid has gaps up to
1.8e15 Hz, and the `active_idx = alphas > 0` selection means the threshold point carries a
weight straddling half a gap into the α=0 region, smearing the edge discontinuity.
**Class B, MEDIUM — recorded in F-005 as a resolution concern, not corrected** (correcting it
properly requires per-continuum subgrids, a larger change than this audit's remit).

**F-006 (class B, HIGH):** `lte_ratios_photoionization` is computed **once**, outside the Λ
loop (`main.py:82-86`), from `lte_populations`, which were evaluated at `ne_bg`. But
`solve_SEE` updates `atmosphere.ne` every iteration (`atoms.py:649`) — measured up to a 93%
change in a single call. Since (n_i/n_k)* ∝ n_e exactly, the recombination rate is then wrong
by exactly the factor n_e/n_e,bg, while the collisional rates in the same matrix use the
*current* n_e. The converged state is not the solution of the coupled problem. The fix is a
one-line rescale by `atmosphere.ne[k]/atmosphere.ne_bg[k]`. Same defect applies to
`bg_species` (frozen H⁻, H I, H II densities) used by the background opacities.

**`hnu_grid[hnu_grid == 0] = 1e-100` (`main.py:71`):** investigated — the grid minimum is
7.396e13 Hz and there is no zero or negative frequency. The guard is **dead code**. It could
only trigger if a continuum's `get_wavelength_grid` produced an infinite wavelength.

---

## 2.6 Collisional rates — **CORRECT**

$$C_{ij}=\frac{8.63\times10^{-6}}{g_j\sqrt T}n_e\Omega_{ij},\qquad
C_{ji}=C_{ij}(n_i^*/n_j^*)\ \text{(detailed balance imposed)}$$

`atoms.py:694-748`. **Detailed balance is imposed, not independently evaluated**, in every
branch ✔:
- `Omega`: C_ji = C0·n_e·Υ/(g_j√T) with C0 = 8.6291e-6 (standard CGS value ✔), then
  C_ij = C_ji(g_j/g_i)e^{−ΔE/kT} ✔.
- `CE`: C_ji = coeff·n_e·(g_i/g_j)√T·1e6, C_ij from detailed balance ✔. The 1e6 converts RH's
  m³ s⁻¹ K^(−1/2) tabulation to cm³ ✔ (matches RH `collision.c`).
- `CI`: C_ij = coeff·n_e·√T·e^{−ΔE/kT}·1e6, C_ji = C_ij·(n_i*/n_j*) ✔ (RH convention).
- `CH`, `CP`: present but **unused by the current configs**. Both multiply by `nh` = *total*
  hydrogen; `CH` should use the neutral-H density and `CP` the proton density.
  **F-011, class A, LOW** (latent — wrong only if a config activates them).

Temperature interpolation: `np.interp` — **linear in T, linear in rate, clamped flat at table
edges** (`atoms.py:705`). RH interpolates linearly in T as well ✔.
**F-005 (class B, MEDIUM):** the H tables span 3000–30000 K but the model reaches 1e5 K, so
the H collision rates are **flat-extrapolated at 14 of 82 depths** (the transition region).
Ca II tables span 3000–1e5 K and are fully covered ✔.

Unknown collision types print a warning and are skipped (`atoms.py:743-745`) — acceptable, but
silent in the sense that the run continues with a physically incomplete model.
Van Regemorter is **not** implemented; no fallback exists for permitted transitions lacking
tabulated data. Not needed by the current configs.

**Verified by Test C:** with J=B the SEE returns Saha–Boltzmann to 1e-5 (H) / 5e-8 (Ca II) —
which cannot happen unless collisional detailed balance is exact.

---

## 2.7 Formal solver — **CORRECT** (the audit brief's stated convention is the one in error)

$$I_O=I_Me^{-\Delta\tau}+\Psi_MS_M+\Psi_OS_O$$

Deriving from I_O = I_M e^{−Δτ} + ∫₀^{Δτ}S(t)e^{−t}dt with t measured **back from O toward M**
and S linear in t:
$$\Psi_M=\frac{w_1}{\Delta\tau},\qquad \Psi_O=w_0-\frac{w_1}{\Delta\tau},
\qquad w_0=1-e^{-\Delta\tau},\ w_1=w_0-\Delta\tau e^{-\Delta\tau}$$

`psi_lin` (`formal_solver.py:958-959`) implements
`psi_m = (1−e^{−Δτ}(1+Δτ))/Δτ` = w₁/Δτ ✔ and `psi_o = (e^{−Δτ}+Δτ−1)/Δτ` = w₀ − w₁/Δτ ✔.

> **Correction to the mission brief:** the brief states Ψ_O = w₁/Δτ and Ψ_M = w₀ − w₁/Δτ,
> which has M and O interchanged. That convention gives I_O → S_M as Δτ→∞, which is
> unphysical; the code's convention correctly gives I_O → S_O. The code is right.

**Catastrophic cancellation: absent.** `psi_lin` uses an 8th-order Taylor branch for
Δτ ≤ 0.1 (`formal_solver.py:951-956`). Verified against reference values over
Δτ ∈ [1e-10, 1e2]: ψ_m, ψ_o correct to full precision and ψ_m+ψ_o+e^{−Δτ} = 1 to machine
precision at every Δτ. `expm1` is **not** needed. Seed lead refuted.

**Test E (linear source, Eddington–Barbier exact):** S = a + bτ ⇒ I(0,μ) = a + bμ.
Measured relative error: 7.8e-15 (μ=1), 6.7e-16 (μ=0.8), 4.0e-15 (μ=0.5), 1.2e-15 (μ=0.2),
3.0e-14 (μ=0.1). **Exact.**

**Test F (constant source):** I = S(1−e^{−τ}) + I₀e^{−τ} reproduced to ≤2.9e-14 for
τ_tot ∈ {1e-8, 1e-4, 1e-2, 1, 10}. **Exact, no cancellation at small τ.**

Interpolation order: **linear** short characteristics. Not parabolic, so there is no
Auer–Paletou overshoot problem to check; the cost is O(Δτ²) accuracy at sharp opacity
gradients versus BESSER/cubic-Hermite in RH and LW. **Class C, documented** — a known,
legitimate, lower-order choice.

Δτ from the trapezoidal mean of χ (`formal_solver.py:914`) — standard, consistent with the
linear-S assumption. ✔

**Sign of dz and direction logic (`main.py:234-237`):** for downward rays
dz = z[iz+1] − z[iz] > 0; for upward rays dz = z[iz] − z[iz−1] > 0; `formal_solution` takes
`np.abs(dz/ray)`. Combined with the loop bounds (upward: iz 0→N−1; downward: iz N−1→0), both
directions are correct. Verified independently for μ>0 and μ<0 in Test E/H. ✔

---

## 2.8 Boundary conditions — **correct at the top; non-standard at the bottom but negligible here (F-008)**

- **Top:** I⁻ = 0 (`main.py:222`) ✔ — no incident radiation, standard.
- **Bottom:** I⁺ = B_ν(T_bottom) (`main.py:218,474`) instead of the diffusion approximation
  I⁺ = B_ν + μ dB_ν/dτ_ν used by RH, LW and SNAPI. **Class C/B.**

  **Quantified:** the emergent-intensity error from omitting the μ dB/dτ term is
  ≈ (μ dB/dτ)·e^{−τ_tot/μ}. Measured total optical depth of this model (LTE populations,
  bottom→top) across the whole 936-point grid: **minimum τ = 20.2** (at 367 nm), with
  τ = 34 at 500 nm, 23 at 400 nm, 4.6e4 at 656 nm. **No point of the spectrum has τ < 5.**
  With e^{−20} = 2e-9, the boundary error is below 1e-8 relative — **negligible for this
  atmosphere**. A controlled test with τ_bot = 8 gives −0.022% at μ=1 and <1e-5 at μ≤0.5.

  Conclusion: the seed lead is real as a *practice* issue but has **no measurable impact on
  this model**. Implemented anyway (F-008, LOW) because it costs three lines and makes the
  code correct for optically thinner models, where the plain-B choice is badly wrong.

- **Λ\* at the boundary node forced to zero (`main.py:231`): CORRECT and consistent.**
  At the starting node the intensity is *prescribed*, so ∂I/∂S_local = 0 exactly, which is
  what the formal solver does there (it never calls `formal_solution` at that node). The
  other ray direction supplies the non-zero Λ* contribution at that depth. No inconsistency.
  ✔ Seed lead refuted.

**Test G (Λ\* verification):** perturbing S at depth k by 1e-6 and re-running the formal
solution gives ∂J_k/∂S_k matching Λ*_kk to **1.1e-10 relative** at k = 5, 20, 40, 55.
Λ* is exactly the diagonal of the operator the formal solver applies. ✔

---

## 2.9 Angular quadrature — **CORRECT**

$$J_\nu=\tfrac12\int_{-1}^{1}I_\nu(\mu)d\mu$$

`get_angular_quadrature_1D` returns `np.polynomial.legendre.leggauss(n)` — nodes on **[−1,1]**
with **Σw = 2.0** (verified). `0.5·w[ir]·I` (`main.py:303,305,307`) is therefore the correct
½∫dμ. **No factor-of-2 error.** ✔

- Odd `n_gauss` is bumped to even (`atmosphere.py:22-24`), so **no μ = 0 node can exist**
  (verified: μ = ±0.2386, ±0.6612, ±0.9325 for n=5→6). `if ray > 0` classifying by μ value is
  therefore safe. ✔
- `if ray == np.max(rays)` (`main.py:309`) selects μ = 0.9325, **not** μ = 1. This is used
  *only* for the per-iteration debug plot; the delivered spectrum comes from the separate
  genuine μ=1 pass at `main.py:484-498`. Cosmetic mislabelling — the plot title says "most
  vertical ray", which is accurate. No finding beyond documentation.

**Test H (isothermal, optically thick, static slab):** J/B = 1.00000000 at mid-slab and at the
bottom, J/B = 0.50000000 at the surface — **machine precision**. This is the decisive test for
quadrature normalization and it passes.

Ray-count convergence is deferred to Test K (Phase 3).

---

## 2.10 Background opacity and scattering — **correct in form, lagged in time (F-007)**

Included (`formal_solver.py:428-449`, Wittmann/Mihalas suite):
H bf (8 levels, Coulomb) + H ff with Gaunt factors; H⁻ bf + ff; H₂⁺; He I bf (10 transitions)
+ ff; He II; He⁻ ff; cool metals C I, Mg I, Al I, Si I, Fe I (T<12000 K); "Luke" metals
N I, O I, Mg II, Si II, Ca II (T<30000 K); Thomson σ_T n_e = 0.6653e-24·n_e ✔;
Rayleigh off H I, He I, H₂.

**F-017 (class A, CRITICAL) — found later, during the benchmark.** The background suite
duplicates the bound-free transitions of any ACTIVE atom: `_opac_h_hydrogenic` re-adds the
active H atom's continua and `_opac_metals_luke` re-adds Ca II's. Measured, the duplicate
equals the original to within 3% across the Lyman continuum — the opacity there was nearly a
factor 2 too large. This was the dominant error in the chromospheric ionization balance. See
FINDINGS F-017. It is recorded here because this section originally enumerated the background
sources without asking whether any of them overlapped the active atoms — the enumeration was
correct as far as it went, and still missed the defect.

**The critical question — is coherent scattering treated as scattering? YES.**
`formal_solver.py:453-456`:
```
opp = kappa + sigma
ems = kappa*B + sigma*atmosphere.J_nu[iz, :]
```
η = κB + σJ, **not** (κ+σ)B. ✔ The continuum source function is correct.

**F-007 (class B, MEDIUM):** `atmosphere.J_nu` is assigned only *after* the ray loop
completes (`main.py:312`), so the scattering emissivity uses the **previous iteration's** J.
This is ordinary Λ-iteration on the scattering term — it converges, but slowly wherever
σ/(κ+σ) → 1 (the near-UV/violet, exactly where Ca II H&K live). It is *inside* the iteration
(not frozen), so it is a convergence-rate defect, not a wrong fixed point. Neither is it
included in the MALI Λ* operator, as RH does. No Ng acceleration exists either.

The docstring at `formal_solver.py:404-406` ("Scattering is treated as absorption with
S_nu = B_nu") **contradicts the code** — stale, see F-009.

`add_background_opacity_old` (`formal_solver.py:666-723`) is dead code (F-009).

Numerical-floor audit: every `np.maximum(..., 0.0)` in the Wittmann ports guards against
negative values from polynomial fits evaluated outside their fitted range. These are inherited
from the original Fortran and are appropriate. Measured: 0 negative opacity samples over the
full depth × frequency grid.

---

## 2.11 LTE populations, ionization, electron density — **CRITICAL BUG (F-001)**

`compute_lte_populations` (`atmosphere.py:86-175`) implements Saha–Boltzmann per stage:
$$\frac{N_{s+1}}{N_s}=\frac{U_{s+1}}{U_s}\frac{2}{n_e}
\Big(\frac{2\pi m_ek_BT}{h^2}\Big)^{3/2}e^{-\chi/k_BT},\qquad
n_i=N_s\frac{g_i}{U_s}e^{-E_i^{\rm rel}/k_BT}$$
The algebra is correct ✔.

**F-001 (class A, CRITICAL).** `atmosphere.py:148-149`:
```
UI, UII, UIII = get_partition_functions(atom.Z, T)
U_t = {0: UI, 1: UII, 2: UIII}
```
For every element except hydrogen, `get_partition_functions` returns (U_neutral, U_singly,
U_doubly) and the mapping is right (verified for Ca: ionization 1→UII, 2→UIII ✔).
**For Z=1 the FIRTEZ table's triplet is (U(H⁻), U(H I), U(H II)) = (1.0, 2.0, 1.0)** — this is
unambiguous from `chemeq.py`'s own tables (`XI[0] = 0.754 eV` is the H⁻ electron affinity,
`XII[0] = 13.595 eV` is the H I ionization potential; `compute_background_species:578-586`
explicitly overrides with `UI_H = 2.0, UII_H = 1.0` and a comment saying so).

So H I receives U(H⁻)=1 instead of 2, and H II receives U(H I) instead of 1.

**Measured impact:**
- Σᵢnᵢ(H)/N_total = **2.000** at cool depths (U off by exactly 2) falling to **0.0040** at the
  1e5 K top (where the excited-state-corrected U(H I) = 249 is applied to the bare proton).
- Test C: with J = B forced, `solve_SEE` returns populations whose **shape** matches
  Saha–Boltzmann to 1.1e-5, but whose **raw** ratio to `lte_populations` is **248** — because
  the SEE conservation row correctly enforces Σn = N_total while the LTE reference does not.
  The SEE is right; the LTE reference is wrong. Ca II by contrast: raw 7.1e-4, shape 5.2e-8.

**What is corrupted:** (a) every hydrogen departure coefficient b_i = n/n_LTE — the headline
deliverable — by factors 0.004 to 248; (b) the initial guess `populations = lte_populations`;
(c) `lte_total_charge` (`atoms.py:594`), the baseline of the Δcharge n_e update, by up to
249× — which is precisely why the `1e-6·nh` electron floor at `atoms.py:630-633` was needed.
**What is not corrupted:** all level *ratios*, hence `lte_ratios_photoionization`, the CI/CH
detailed-balance ratios, and the bf emissivity ratio — U cancels exactly in all of them.

**n_e treatment.** `pel` from the config is parsed and then **ignored**; n_e is solved from a
92-element LTE charge balance on (T, p_g) (`chemeq.compute_background_eos`). It is then
updated each Λ-iteration from the active atoms' Δcharge with 0.5 damping and the metal floor.
NLTE populations therefore do feed back into n_e (RH-like), but `lte_populations`,
`lte_ratios_photoionization` and `bg_species` are never re-evaluated at the updated n_e —
see F-006. Whether SNAPI updates n_e at all is an open question (`OPEN_QUESTIONS.md`).

**Σᵢnᵢ = A_elem n_H conservation:** enforced exactly by the SEE conservation row at every
depth and iteration (verified: Σn/N_total = 1.000000 for both atoms after `solve_SEE`). ✔
Note that for Ca II the model has no Ca I stage, so all Ca is distributed over Ca II + Ca III;
this is a deliberate model choice matching the SNAPI atom (class C).

---

## 2.12 MALI / accelerated Λ-iteration — **CORRECT preconditioning, defensive clamps (F-002 knock-on)**

Correct preconditioned rates (Rybicki & Hummer 1991/1992; RH Uitenbroek 2001 §3). Substituting
J̄ = J̄_eff + Λ*S_l with S_l = n_jA_ji/(n_iB_ij − n_jB_ji) into n_iR_ij − n_jR_ji and using
(n_iB_ij − n_jB_ji)S_l = n_jA_ji:
$$R_{ij}=B_{ij}\bar J_{\rm eff},\qquad R_{ji}=A_{ji}(1-\Lambda^*)+B_{ji}\bar J_{\rm eff}$$

`atoms.py:778-779`:
```
R_matrix[i, j] += line.Blu * J_eff
R_matrix[j, i] += line.Aul * (1.0 - L_star) + line.Bul * J_eff
```
**Exactly the Rybicki–Hummer form.** The S_l dependence on (n_i, n_j) is moved analytically to
the left-hand side; this is **not** the "substitute J̄ − Λ*S into the old rates" shortcut the
brief warns about. ✔

- Λ* is the **diagonal** local operator (Olson–Auer–Buchler), and Test G proves it is the
  diagonal of the *same* operator the formal solver applies (agreement 1.1e-10). ✔
- The Λ̄* line-average correctly weights by the opacity ratio χ_line/χ_total
  (`main.py:297-305`), which is ∂S_total/∂S_l — the right chain-rule factor. ✔
- `S_old` uses `old_pops_for_S`, the populations from the **outer** Λ-iteration, not the
  locally-updated ones (`atoms.py:589-595, 611-612`) ✔ — correct MALI.
- **Reduces exactly to ordinary Λ-iteration when Λ*≡0**: with L_star = 0,
  J_eff = max(J̄, 0) = J̄ and the rates become B_ij J̄ and A_ji + B_ji J̄. ✔ (Test J, Phase 3.)

**Three clamps that can move the fixed point:**
1. `L_star = np.clip(Λ̄*, 0.0, 0.9999999)` (`atoms.py:760`). Mathematically Λ̄* ≤ 1 always,
   since Σwφ_norm = 1, ψ_o ≤ 1 and χ_line/χ_tot ≤ 1 — so this can only fire when the profile
   normalization is broken (F-002) or under masing (χ_line < 0). Defensive, but it is masking
   an F-002 symptom.
2. `J_eff = max(J̄ − Λ*S_old, 0.0)` (`atoms.py:775`). J̄_eff should be ≥0 in a consistent
   scheme; it goes negative when Λ̄* overshoots — again an F-002 symptom. **This clamp does
   change the converged solution when it fires**, and it is the one to watch after F-002 is
   fixed.
3. `populations_new[populations_new < 0] = 1e-100` (`atoms.py:815`).

**Convergence criterion (class C, MEDIUM):** measured only on populations and n_e
(`atoms.py:654-675`), with a default tolerance of 1e-2 in `config_H_Ca_Mg_Na.json` — very
loose. No criterion on J̄, and the SE residual ‖P·n‖ is never reported. **No Ng acceleration**
(performance gap, not a correctness bug, as the brief notes).

---

## 2.13 Broadening recipes — **ALL VERIFIED CORRECT**

Each recipe was checked term by term against Lightweaver's `Broadening.py` / RH's `broad.c`,
including a dimensional analysis.

**Unsold** (`main.py:116-130`). Γ/N_H = 8.08 v̄^(3/5) C₆^(2/5).
- `C6 = 2.5 e² α_H 2π(Z a₀)²/h · |ΔR̄²|` — CGS translation of LW's
  `2.5 Q²/(4πε₀) · ABarH/(4πε₀) · 2π(Z R_Bohr)²/h`. Units: [erg cm][cm³][cm²]/[erg s] = cm⁶ s⁻¹ ✔
  (the correct C₆ unit for ΔE/ħ = C₆/R⁶).
- `vRel35H = (8k_B/(π m_A)(1 + m_A/m_H))^0.3`. Since 8k_BT/(πμ) = (8k_BT/(π m_A))(1+m_A/m_H)
  = v̄², raising to 0.3 gives v̄^0.6 = v̄^(3/5) ✔ — the reduced-mass factor is exactly right.
- `T**0.3` applied at use (`main.py:270`) completes v̄^(3/5) ∝ T^0.3 ✔.
- Dimensional check: [cm/s]^0.6 [cm⁶/s]^0.4 [cm⁻³] = cm³ s⁻¹ · cm⁻³ = **s⁻¹** ✔.
- He term: `vals[1]·he_abund·vRel35He` multiplied by `nHGround` gives Γ_He ∝ n_He^eff — the
  same approximation LW makes (n_He ≈ he_abund × n_H(ground) rather than the true He I
  density). **Class C, documented.** `he_abund` is a *number ratio* (0.1), not a mass
  fraction ✔ — the seed concern is unfounded.
- Z = ionization + 1 ✔ (Z=1 for H I, Z=2 for Ca II — correct effective charge).

**Barklem/ABO** (`atoms.py:116-126`).
$\Gamma/N_H=(4/\pi)^{\alpha/2}\Gamma\!\big(\tfrac{4-\alpha}{2}\big)v_0\sigma(\bar v/v_0)^{1-\alpha}$
- `Γ(2 − α/2)` = Γ((4−α)/2) ✔.
- `crossSection = σ·a₀²·(v̄/1e6)^(−α)` — σ in **a₀²** ✔, and v₀ = 10⁴ m s⁻¹ = **10⁶ cm s⁻¹** ✔.
- `meanVel·crossSection` = σ_eff v̄ (v̄/v₀)^(−α) = v₀σ(v̄/v₀)^(1−α) ✔.
- **Factor 2** (`result[0] = 2.0 * …`): converts the ABO half-width to a full width, matching
  LW's `VdwBarklem` ✔. Consistent with a = Γ_full/(4πΔν_D).
- `meanVel` omits T; the T dependence enters at use as `T**(0.5*(1−α))` (`main.py:276`) ✔,
  since v̄ ∝ T^0.5 and Γ ∝ v̄^(1−α).
- `reducedMass = m_u/(1/1.008 + 1/mass)` ✔ correct reduced mass in amu.
- Table bounds are hard-enforced with a `ValueError` (`atoms.py:108-111`) rather than
  extrapolated ✔ — good practice.

**Quadratic Stark** (`main.py:156-179`). Γ = 11.37 C₄^(2/3) v̄^(1/3) N_pert.
- `11.37` = the Lindholm coefficient ✔.
- `C4 = e² a₀ (2π a₀²/h)/(18 Z⁴)·|(n_u(5n_u²+1))² − (n_l(5n_l²+1))²|` — Traving (1960) via
  RH/LW ✔. Units [erg cm][cm][cm²]/[erg s] = cm⁴ s⁻¹ ✔.
- `Cm = (1+m_A/m_e)^(1/6) + (1+m_A/28m_u)^(1/6)`: v̄^(1/3) ∝ μ^(−1/6), so these are the
  electron and singly-ionized-perturber reduced-mass factors ✔. The hard-coded **28.0 average
  atomic weight** for the ionic perturbers is exactly what RH and LW use — **class C,
  documented**. Assuming n_ion ≈ n_e is likewise standard ✔.
- n_eff = Z_i√(E_Ryd^elem/(E_cont − E_level)) with the reduced-mass-corrected Rydberg
  (`main.py:169`) ✔ and Z_i = lower stage + 1 ✔.
- Dimensional check: [cm⁴/s]^(2/3)[cm/s]^(1/3)[cm⁻³] = **s⁻¹** ✔.

**Linear Stark, hydrogen** (`main.py:181-190`, Sutton 1978).
`lin_stark_factor = a1·0.6·(n_u²−n_l²)·4π·0.425`, applied as `× n_e^(2/3)`.
**The "10⁻⁴ scalar" comment is CORRECT — explicit dimensional analysis:** Lightweaver
evaluates `… × 1e-4 × n_e[m⁻³]^(2/3)`. With n_e[m⁻³] = 10⁶ n_e[cm⁻³],
(10⁶)^(2/3) = **10⁴** exactly, so 10⁻⁴ × (10⁶ n_e^CGS)^(2/3) = n_e^CGS^(2/3).
Dropping the 10⁻⁴ when passing n_e in cm⁻³ is exactly right. ✔ Seed lead resolved — not a bug.
n_upper/n_lower from `round(√(g/2))` ✔ (g = 2n² for the merged-l hydrogen levels).

**Natural.** Γ_rad is **read from the config** (`main.py:262-264`), not computed as
Σ_{k<j}A_jk + Σ_{k<i}A_ik. Cross-check against the Einstein coefficients of the parsed model:

| atom | i→j | λ₀ (nm) | γ_config | Σ_k A_jk | Σ_k A_ik | correct total | γ_cfg / correct |
|---|---|---|---|---|---|---|---|
| H | 0→1 Lyα | 121.57 | 4.700e8 | 4.696e8 | 0 | 4.696e8 | **1.001** |
| H | 0→2 Lyβ | 102.57 | 9.980e7 | 9.979e7 | 0 | 9.979e7 | **1.000** |
| H | 1→2 Hα | 656.47 | 9.980e7 | 9.979e7 | 4.696e8 | 5.694e8 | **0.175** |
| H | 1→3 Hβ | 486.27 | 3.020e7 | 3.017e7 | 4.696e8 | 4.998e8 | **0.060** |
| H | 1→4 Hγ | 434.17 | 1.150e7 | 1.155e7 | 4.696e8 | 4.812e8 | **0.024** |
| H | 2→3 Paα | 1875.63 | 3.020e7 | 3.017e7 | 9.979e7 | 1.300e8 | **0.232** |
| H | 2→4 | 1282.17 | 1.150e7 | 1.155e7 | 9.979e7 | 1.113e8 | **0.103** |
| H | 3→4 Brα | 4052.29 | 1.150e7 | 1.155e7 | 3.017e7 | 4.172e8 | **0.276** |
| Ca II | 0→3 H | 396.96 | 1.480e8 | 1.550e8 | 0 | 1.550e8 | 0.955 |
| Ca II | 0→4 K | 393.48 | 1.500e8 | 1.577e8 | 0 | 1.577e8 | 0.951 |
| Ca II | 1→3, 1→4, 2→4 (IR) | 850–866 | 1.48–1.50e8 | 1.55–1.58e8 | **0** | — | 0.951–0.955 |

The config γ reproduces **Σ_k A_jk exactly** (the upper level's total radiative width) and
omits Σ_k A_ik entirely.
- **Resonance lines are correct** (lower level = ground ⇒ Σ_k A_ik = 0): H Lyman series
  ratio 1.000–1.001 ✔.
- **Ca II is correct throughout** ✔ — the lower levels of the IR triplet are the metastable
  3d levels, which have no permitted downward transition in the model (3d→4s is forbidden),
  so Σ_k A_ik = 0 genuinely. The residual 4.5–4.9% is a minor inconsistency between the
  config's γ and the A-values implied by its own f-values, not a missing term.
  **The SNAPI benchmark is therefore unaffected.**
- **Hydrogen subordinate lines are under-specified by 3.6× to 42×** (Hγ worst at 0.024).

**F-010, class A, MEDIUM.** Impact assessment: for these lines the linear Stark term dominates
the damping wherever n_e is appreciable — for Hα at the model bottom Γ_Stark ≈ 2.5e11 s⁻¹
versus a natural-width error of 4.7e8 s⁻¹, i.e. irrelevant. The error only becomes significant
in the upper chromosphere/transition region where n_e^(2/3) collapses (at the model top
Γ_Stark ≈ 5.4e7 s⁻¹, comparable to the missing 4.7e8 s⁻¹). This is a **data** defect, not a
code defect — the code faithfully uses what it is given. Recorded and **not patched**: editing
atomic data is outside the audit's remit, and doing so would decouple the model from the
reference atom files. The remedy, if wanted, is to compute Γ_rad from the model rather than
read it (which is what a Σ_k A_jk + Σ_k A_ik helper would provide).

---

## 2.14 Redistribution — **class C, documented (F-012)**

The code is **pure CRD** (ψ = φ everywhere). The `type` field in the config ("PRD"/"CRD") is
parsed into `Line.type` and **never read**. Lines flagged PRD in `config_H_Ca_Mg_Na.json`:
H Lyα, H Lyβ, Ca II H (396.96 nm), Ca II K (393.48 nm).

RH, Lightweaver and SNAPI all offer angle-averaged PRD (Hummer 1962) or hybrid PRD
(Leenaarts et al. 2012) for exactly these transitions. The expected error from CRD:
- **Ca II H & K:** CRD overestimates the wing source function, filling in the inner wings and
  raising the emergent intensity at ~0.1–0.5 nm from line centre by tens of per cent; the
  characteristic PRD emission-peak asymmetry is absent. This will **dominate the SNAPI
  comparison** for H & K if SNAPI ran in PRD mode.
- **Lyα/Lyβ:** similar, larger.
- **Ca II infrared triplet (854.2, 866.2, 850.0 nm): CRD is the correct choice** — these are
  formed under near-complete redistribution and RH/LW/SNAPI treat them in CRD too. The IR
  triplet is therefore the clean channel for the benchmark.

**Not implemented** (the brief says not to). Documented prominently, and flagged as the first
thing to check in the SNAPI setup reconciliation.

---

## 2.15 Code hygiene — F-009 (LOW, filed but deferred behind the physics)

| Issue | Location | Effect |
|---|---|---|
| `eval(configuration["debug"])` vs `configuration.get("debug", False)` | `main.py:40` vs `534` | The JSON value is the **string** `"True"`. `eval("True")` → True; `.get(...)` returns the truthy string `"True"`. Both happen to be True here, but a JSON `"False"` would give False at L40 and **True** at L534 — the two gates disagree. |
| Config filename hardcoded; `sys.argv` ignored | `main.py:14` | The SNAPI comparison pipeline passes a config path that is silently discarded — **the benchmark has never run the config it claims** (F-013). |
| `tau_depth[0,:]` set to a non-zero value under a comment claiming "zero" | `main.py:476` | The bottom row of the optical-depth array is offset by χ(0)·Δz, biasing every saved τ scale. |
| `if 'emergent_I_vertical' in locals()` | `main.py:366` | Fragile; silently skips plots. |
| Diagnostic plotting inside the Λ loop | `main.py:345-442` | Dozens of figures per iteration; dominates runtime. |
| `add_background_opacity_old` dead code | `formal_solver.py:666-723` | 60 lines of unreachable duplicate physics. |
| Stale docstring contradicting the code | `formal_solver.py:404-406` | Claims scattering is treated as κB; the code correctly uses σJ. |
| `get_RT_coefficients` depends on attributes injected by `main.py` | `formal_solver.py:334` etc. | `line.vdw_cross`, `line.stark_c23`, … are set as side effects in `main.py:94-193`, so the RT module **cannot be exercised without running main.py** — the direct cause of there being no unit tests for it. |
| Voigt/damping block duplicated verbatim | `main.py:259-290` vs `formal_solver.py:323-355` | Divergence hazard; also doubles the cost of the most expensive kernel. |
| `voigt()` is a Python loop over frequencies | `formal_solver.py:273` | ~1e4 calls/iteration; `scipy.special.wofz` is vectorised, exact, and ~100× faster. |
