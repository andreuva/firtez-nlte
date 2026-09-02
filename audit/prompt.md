## ROLE

You are a senior numerical-radiative-transfer engineer with deep working knowledge of the
non-LTE line formation codes used in solar physics: **RH** (Uitenbroek 2001), **RH 1.5D**
(Pereira & Uitenbroek 2015), **Lightweaver** (Osborne & Milić 2021), **SNAPI** (Milić & van
Noort 2018), and **MULTI** (Carlsson 1986). Your standard of correctness is
*Hubeny & Mihalas, Theory of Stellar Atmospheres* (2014) — hereafter **HM2014** — together with
Mihalas (1978) and Rutten's *Radiative Transfer in Stellar Atmospheres*.

Your job is **not** to make the code run or to make plots look nicer. Your job is to determine
whether every physical equation and every numerical scheme in this code is *analytically and
numerically correct*, to prove it with tests, and to fix what is not. After your work,
the code should give accurate spectral synthesis and departure coefficients for the computed atoms.

## MISSION

Audit the NLTE synthesis code in this repository — `main.py` plus **all** submodules
(`constants.py`, `atmosphere.py`, `atoms.py`, `formal_solver.py`, and anything they import),
plus the JSON configuration files (`config_H_Ca_Mg_Na.json` and siblings) and the atomic data
files they reference.

Benchmark data lives at:

```
/home/andreuva/Documents/firtez-nlte/tests/compare_snapi
```

Deliver a prioritized findings register, minimal surgical patches, and a regression test suite
that proves each fix.

## RULES OF ENGAGEMENT

1. **Read before you write.** Do not modify a single line during Phase 0–1.
2. **One finding, one patch, one test.** Never bundle unrelated changes into a commit.
3. **Cite your standard.** Every claim of "this is wrong" must reference a specific equation
   (HM2014 section/eq. number, a paper, or the corresponding routine in RH/Lightweaver). "This
   looks off" is not a finding.
4. **No fudge factors.** You may not introduce an empirical scaling, a tuned coefficient, or a
   clamp to make output match the benchmark. If output only matches after a magic number, the
   physics is wrong — say so.
5. **Clipping is a smell, not a fix.** The code is full of `np.maximum(x, 1e-100)` and
   `np.clip`. Treat each as a potential mask over a real bug. For each one, determine *why* the
   quantity could reach that limit. Never add new ones without justification.
6. **Distinguish the three failure classes** in every finding:
   - **(A) Wrong physics** — the implemented equation differs from the correct one.
   - **(B) Correct physics, wrong numerics** — discretization, quadrature, cancellation, grid.
   - **(C) Correct but non-standard** — a legitimate approximation (e.g. CRD instead of PRD)
     that departs from RH/Lightweaver/SNAPI and must be documented, not silently kept.
7. **Preserve the interface.** Do not restructure the module layout, rename public functions, or
   introduce new dependencies without asking first. Prefer small diffs.
8. **Determinism.** Fix random seeds; make every test reproducible from a clean checkout.
9. If you cannot determine correctness from the code alone, **say so explicitly** and state what
   information would settle it. Do not guess and present the guess as a conclusion.

---

## PHASE 0 — INVENTORY (no edits)

Produce `audit/00_inventory.md` containing:

- A call graph from `main.py` down through every function that touches physics.
- For every array carrying a physical quantity: name, shape, **units**, indexing convention
  (depth ordering: is index 0 the bottom or the top?), and where it is first assigned.
- The atomic model as actually parsed: levels (energy, `g`, ionization stage), lines
  (`nu0`, `Aul`, `Blu`, `Bul`, `max_delta_nu`), continua (thresholds, cross-section source),
  and all collisional data with its temperature interpolation scheme.
- The atmosphere: is `zgrid` increasing upward or downward? Is `ne` given or solved? Is there a
  velocity field, and does anything use it?
- What is fixed vs iterated: confirm exactly which quantities are recomputed inside the Λ loop.

## PHASE 1 — UNITS & CONVENTIONS REGISTER (no edits)

Most NLTE bugs are unit and convention bugs. Produce `audit/01_units.md`: a table mapping each
symbol to its code variable, expected CGS units, and the actual units implied by how it is used.
Check specifically:

- Per-frequency vs per-wavelength: is `I` in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ *everywhere*?
- Energies: erg vs eV vs cm⁻¹ vs J. `E_Ryd_erg`, level energies, `E_cont`.
- Angular frequency vs ordinary frequency — every stray or missing `2π` and `4π`.
- Damping constants: is `gamma` a full width in s⁻¹, an angular rate in rad s⁻¹, or a HWHM?
  This determines whether `a = Γ/(4π Δν_D)` is right.
- Cross-sections: cm², a₀², or Mb (10⁻¹⁸ cm²)?
- Number densities: cm⁻³ throughout; abundances as number ratios relative to H or to totals.
- `atmosphere.he_abund` — number ratio He/H, or mass fraction? Used as a bare multiplier in the
  van der Waals terms; verify.

**Flag every place a factor of 2, 4π, or π is applied and state its origin.**

---

## PHASE 2 — EQUATION-BY-EQUATION AUDIT

For each item below, write in `audit/02_physics.md`: (i) the correct equation in LaTeX,
(ii) the code as implemented with `file:line`, (iii) the difference, if any, (iv) the failure
class (A/B/C), (v) severity. Work through **all** of these — do not stop at the first bug.

### 2.1 Statistical equilibrium

Correct form (HM2014 §14.2):

$$\sum_{j\neq i} n_j P_{ji} - n_i \sum_{j\neq i} P_{ij} = 0, \qquad P_{ij} = R_{ij} + C_{ij}$$

closed by particle conservation $\sum_i n_i = n_{\rm elem}$, which must **replace** one row —
conventionally the row of the most populated level at that depth, for conditioning.

Verify: matrix assembly and sign convention; that the replaced row is chosen per depth point,
not fixed; that the ionized-stage/continuum level is included in the conservation sum; that the
solve is not silently producing negative populations; the conditioning of the matrix and whether
a scaling/equilibration is applied before the solve.

### 2.2 Bound–bound radiative rates

$$R_{ij} = B_{ij}\bar{J}_{ij}, \qquad R_{ji} = A_{ji} + B_{ji}\bar{J}_{ij}, \qquad
\bar{J} = \oint\frac{d\Omega}{4\pi}\int \phi(\nu,\Omega)\,I(\nu,\Omega)\,d\nu$$

Check the Einstein relations hold in the parsed atomic model to machine precision:
$A_{ji} = (2h\nu^3/c^2)B_{ji}$ and $g_i B_{ij} = g_j B_{ji}$. If the atom file supplies $f$-values,
verify the $f \to A$ conversion including the $g_i/g_j$ ratio.

### 2.3 Line profile and its normalization — **HIGH PRIORITY**

$$\phi(\nu) = \frac{H(a,v)}{\sqrt{\pi}\,\Delta\nu_D}, \quad
v = \frac{\nu-\nu_0 - \nu_0\,\boldsymbol{\mu}\!\cdot\!\mathbf{v}_{\rm los}/c}{\Delta\nu_D},\quad
\Delta\nu_D = \frac{\nu_0}{c}\sqrt{\frac{2k_BT}{m} + \xi^2}, \quad \int\phi\,d\nu = 1$$

In `main.py` the profile is **truncated** (`voigt_line[~mask] = 0.0`, ~line 288) and then
**renormalized numerically** (`voigt_norm = voigt_line / np.sum(voigt_line*weigths_freq_grid)`).
The author has left a TODO asking why this matters so much. Investigate this properly:

- Numerical renormalization silently absorbs *any* error in the frequency grid, the quadrature
  weights, or the analytic $1/(\sqrt{\pi}\Delta\nu_D)$ prefactor. Compute the analytic profile
  and compare $\int\phi\,d\nu$ against 1 **before** renormalizing, at every depth and every line.
  If it is not 1 to within 10⁻⁶, the grid or the weights are wrong — that is the real bug.
- Truncating a Voigt profile at `max_delta_nu` and then renormalizing redistributes the missing
  damping-wing weight into the core, artificially inflating core opacity and $\bar{J}$. Quantify
  the discarded fraction. Compare with how RH handles per-transition wavelength grids and how
  Lightweaver sets `qWing`.
- Verify `voigt()` returns $H(a,v)$ (i.e. `Re[w(v + ia)]` for the Faddeeva function) and not
  some other normalization; test against known values of $H(a,v)$ and against the two limits
  $a\to 0$ (Gaussian) and large $v$ (Lorentzian wing $a/(\sqrt{\pi}v^2)$).
- Check `2` in $2k_BT/m$ (most-probable speed, not RMS) and that $\xi$ is the turbulent velocity,
  not a FWHM.
- **The profile has no line-of-sight velocity term and no $\mu$ dependence.** If the atmosphere
  carries a velocity field, `dop_freq` is wrong for every inclined ray. Confirm and report.

### 2.4 Line opacity and emissivity

$$\chi_\nu^{\rm line} = \frac{h\nu_0}{4\pi}\left(n_i B_{ij}\phi_\nu - n_j B_{ji}\psi_\nu\right),
\qquad \eta_\nu^{\rm line} = \frac{h\nu_0}{4\pi} n_j A_{ji}\psi_\nu$$

with $\psi=\phi$ under CRD. Verify the stimulated-emission term is treated as negative
absorption (it is, at ~line 296) and that `get_RT_coefficients` uses the *identical* profile,
normalization and populations as the $\bar{J}$ integration loop — any inconsistency between the
two breaks the MALI preconditioning. Check whether negative total opacity can occur and what
happens if it does.

### 2.5 Bound–free rates

$$R_{ik} = 4\pi\int_{\nu_0}^{\infty}\frac{\alpha_\nu}{h\nu}J_\nu\,d\nu, \qquad
R_{ki} = 4\pi\left(\frac{n_i}{n_k}\right)^{\!*}\int_{\nu_0}^{\infty}\frac{\alpha_\nu}{h\nu}
\left(\frac{2h\nu^3}{c^2}+J_\nu\right)e^{-h\nu/k_BT}d\nu$$

$$\left(\frac{n_i}{n_k}\right)^{\!*} = n_e\,\frac{g_i}{2g_k}
\left(\frac{h^2}{2\pi m_e k_B T}\right)^{3/2}\exp\!\left(\frac{\chi_{\rm ion}}{k_BT}\right)$$

Verify: the Saha–Boltzmann ratio (`lte_ratios_photoionization`) is a function of $T$ and $n_e$
only — confirm it is genuinely that, and not contaminated by NLTE-updated populations. The
factor $g_k$ must be the statistical weight of the **ground state of the next ion**, with the
factor 2 from the free electron spin. Check $\alpha_\nu \equiv 0$ below threshold, the Gaunt
factor treatment for hydrogenic cross-sections, and the interpolation of tabulated $\alpha$ onto
the global grid (extrapolation must not produce spurious non-zero values). Also verify the
$d\nu$ quadrature weights are the correct ones for the continuum range — reusing line-tuned
weights over a continuum is a classic error.

Note: `hnu_grid[hnu_grid == 0] = 1e-100` implies a zero frequency exists in the grid. Find out
why.

### 2.6 Collisional rates

$$C_{ij} = \frac{8.63\times10^{-6}}{g_i\sqrt{T}}\,n_e\,\Omega_{ij}\,e^{-\Delta E/k_BT},
\qquad C_{ji} = C_{ij}\left(\frac{n_i^*}{n_j^*}\right)^{-1}\!\!\text{(detailed balance)}$$

Verify detailed balance is **imposed**, not independently evaluated; the temperature
interpolation of tabulated $\Omega$/CE/CI data (log-log? linear? clamped at table edges?); the
van Regemorter fallback for permitted transitions if used; collisional ionization; and
collisions with neutral H if present. Confirm which RH/MULTI keywords the parser supports and
what it does with ones it does not.

### 2.7 Formal solver — **HIGH PRIORITY**

$$I_O = I_M e^{-\Delta\tau} + \Psi_M S_M + \Psi_O S_O \;(+\;\Psi_P S_P),\qquad
\Delta\tau = \tfrac{1}{2}(\chi_M+\chi_O)\frac{|\Delta z|}{|\mu|}$$

with, for linear interpolation of $S$,
$w_0 = 1-e^{-\Delta\tau}$, $w_1 = w_0 - \Delta\tau e^{-\Delta\tau}$,
$\Psi_O = w_1/\Delta\tau$, $\Psi_M = w_0 - w_1/\Delta\tau$, and $\Lambda^*_{OO} = \Psi_O$.

Check:

- **Catastrophic cancellation for small $\Delta\tau$.** $w_0$ and $w_1$ must use `expm1` or a
  Taylor expansion below $\Delta\tau \sim 10^{-3}$. This is the single most common silent bug in
  hand-written formal solvers and it corrupts the optically thin upper atmosphere.
- Interpolation order: linear, parabolic (Auer & Paletou 1994), or BESSER/cubic-Hermite
  (Štěpán & Trujillo Bueno 2013). Parabolic without monotonicity control overshoots at sharp
  opacity gradients — check for it.
- Is $\Delta\tau$ built from the *trapezoidal* mean of $\chi$, or should it use a higher-order
  integration on a strongly stratified grid?
- Sign of `dz` and the direction logic for up/down rays (`main.py` ~lines 234–237). Verify
  independently for both directions with a case whose answer is known analytically.

### 2.8 Boundary conditions — **HIGH PRIORITY**

- Top: $I^-=0$ (or a prescribed incident field). Confirm.
- Bottom: the code uses $I^+ = B_\nu(T_{\rm bottom})$ (`main.py` line 218 and line 474). The
  industry standard is the **diffusion approximation**,
  $I^+ = B_\nu(T) + \mu\,\dfrac{dB_\nu}{d\tau_\nu}$, which RH, Lightweaver and SNAPI all use.
  Using plain $B$ under-illuminates the deepest layers, biases the continuum, and shows up as a
  $\mu$-dependent error in the emergent intensity. Quantify the effect and implement the
  diffusion approximation.
- The Λ* at the boundary point is forced to zero (line 231). Verify that this is consistent with
  what the formal solver actually does at that node — an inconsistent $\Lambda^*$ does not just
  slow convergence, it converges to the **wrong** answer if the rates are preconditioned with it.

### 2.9 Angular quadrature

$$J_\nu = \frac{1}{2}\int_{-1}^{1} I_\nu(\mu)\,d\mu$$

The code accumulates `0.5 * weigths[ir] * I_o`. Determine whether
`get_angular_quadrature_1D` returns Gauss–Legendre nodes on $[-1,1]$ (weights summing to 2) or a
hemispheric set. **A factor-of-2 error here is invisible in the LTE continuum and lethal in the
line cores.** Test: an isothermal, optically thick, static slab must return $J_\nu = B_\nu$ to
machine precision.

Also check:
- `if ray > 0` classifies by $\mu$ **value**, which is correct for a symmetric set but silently
  wrong if any $\mu = 0$ node exists.
- `if ray == np.max(rays)` selects the emergent intensity — this is the largest quadrature $\mu$,
  **not** $\mu=1$. Confirm the intent and whether the final $\mu=1$ formal solution supersedes it.
- Whether the number of rays is sufficient for the $\bar{J}$ integral to be converged
  (grid-doubling test in Phase 3).

### 2.10 Background opacity and scattering — **HIGH PRIORITY**

Enumerate what `get_RT_coefficients` includes: H⁻ bound-free and free-free (Wishart 1979; Bell &
Berrington 1987), H bound-free and free-free with Gaunt factors, Thomson scattering
$\sigma_T n_e$, Rayleigh scattering off H I / He I / H₂, metal continua, background line
blanketing.

The critical question: **is coherent scattering treated as scattering?** Its contribution to the
emissivity must be $\eta = \sigma_\nu J_\nu$, not $\sigma_\nu B_\nu$. If scattering is lumped
into thermal emission, the continuum source function is wrong above $\tau \approx 1$ and every
departure coefficient is wrong with it. If scattering *is* included as $\sigma J$, confirm it is
inside the iteration and not frozen at the initial guess.

### 2.11 LTE populations, ionization, electron density

Verify `compute_lte_populations` against Saha–Boltzmann with proper partition functions; check
whether $n_e$ is taken from the input atmosphere or solved from charge conservation, and whether
the NLTE populations of the active atoms feed back into $n_e$ (RH does this optionally; SNAPI
handles it differently — document the choice). Confirm $\sum_i n_i = A_{\rm elem}\,n_{\rm H}$ is
conserved at every depth and every iteration.

### 2.12 MALI / accelerated Λ-iteration — **HIGH PRIORITY**

The code computes `Lambda_star_bar` weighted by an `opacity_ratio` $=\chi_{\rm line}/\chi_{\rm tot}$
(lines 296–305) and passes it to `solve_SEE`. Verify against the preconditioned rate equations of
**Rybicki & Hummer (1991, 1992)** as implemented in RH (Uitenbroek 2001, §3) and Lightweaver:

- Is $\Lambda^*$ the **diagonal** (local) operator of Olson, Auer & Buchler (1986) / Olson &
  Kunasz (1987), and is it the diagonal of the *same* operator the formal solver applies?
- Are the preconditioned rates constructed so that the $S_l$ dependence on $n_i,n_j$ is moved to
  the left-hand side of the SE matrix, or is $\bar{J}-\bar{\Lambda}^*S_l$ merely substituted into
  the old rates? The latter is a common and subtly wrong shortcut.
- Does the scheme reduce **exactly** to ordinary Λ-iteration when $\Lambda^*\equiv 0$? It must.
  Run with `Lambda_star_bar` forced to zero and confirm both converge to the same fixed point
  (slowly, but to the same answer). *If they converge to different populations, the
  preconditioning is wrong* — this is the decisive test.
- Is Ng acceleration implemented? If not, note it as a performance gap, not a correctness bug.
- Convergence is measured only on populations. Add a criterion on $\bar{J}$ or on the SE
  residual, and report the residual $\|\mathbf{P}\mathbf{n}\|$, not just $\max|\Delta n/n|$.

### 2.13 Broadening recipes

Check each against Lightweaver's `Broadening.py` and RH's `broad.c`:

- **Unsold**: $\Gamma/N_{\rm H} = 8.08\,v_{\rm rel}^{0.3}\,C_6^{0.4}$; verify the $C_6$ expression,
  the reduced-mass factors $(1+m/m_{\rm H})$, the He term scaling by the He abundance, and the
  $T^{0.3}$ applied at line 270/276.
- **Barklem/ABO**: $\Gamma/N_{\rm H} = \left(\frac{4}{\pi}\right)^{\alpha/2}
  \Gamma\!\left(\frac{4-\alpha}{2}\right)v_0\,\sigma\,(\bar{v}/v_0)^{1-\alpha}$ with
  $v_0 = 10^4\ {\rm m\,s^{-1}}$. Verify the $\sigma$ units (a₀²), the $T^{0.5(1-\alpha)}$
  exponent, and whether the result is a HWHM or a full width (factor 2).
- **Quadratic Stark**: the $11.37\,(C_4)^{2/3}$ Lindholm coefficient, $C_4$ from Traving (1960),
  the $T^{1/6}$ dependence, the effective quantum numbers $n_{\rm eff}$, and the hard-coded
  average atomic weight 28.0.
- **Linear Stark (H)**: Sutton (1978). Line 190 contains a comment about "dropping the 10⁻⁴
  scalar when taking ne_CGS vs ne_SI" — **verify this unit hand-wave explicitly with a
  dimensional analysis**; it is exactly the kind of thing that is off by 10⁴.
- **Natural**: is $\Gamma_{\rm rad}$ read from config, or computed as
  $\sum_{k<j}A_{jk} + \sum_{k<i}A_{ik}$ (both upper *and* lower level)? Cross-check the config
  values against the sum over the atomic model.

### 2.14 Redistribution

The code is pure CRD. For Ca II H&K, Mg II h&k and Lyα/Lyβ this is a **class C** departure from
RH/Lightweaver/SNAPI, which use angle-averaged PRD (Hummer 1962) or hybrid PRD (Leenaarts et al.
2012). Do not implement PRD unless asked — but quantify the expected error in the wings and
document it prominently, because it will dominate the SNAPI comparison for those lines.

### 2.15 Code hygiene (low severity, report anyway)

`eval(configuration["debug"])` at line 40 vs `configuration.get("debug", False)` at line 534 —
a JSON string `"False"` is truthy, so the two debug gates disagree. Diagnostic plotting inside
the Λ loop. `if 'emergent_I_vertical' in locals()`. `tau_depth[0,:]` initialized with a nonzero
value under a comment claiming it is zero (line 476). Fix these only after the physics findings
are filed.

---

## PHASE 3 — ANALYTIC TEST SUITE (write these **before** touching the benchmark)

Create `tests/test_physics.py` (pytest). Every test must have an answer known in closed form, so
that a failure localizes the bug. **Do not proceed to Phase 4 until these pass.**

| # | Test | Expected result |
|---|------|-----------------|
| A | Profile normalization | $\int\phi\,d\nu = 1$ to $<10^{-6}$, every line, every depth, **before** numerical renormalization |
| B | Einstein relations | $A_{ji}=(2h\nu^3/c^2)B_{ji}$, $g_iB_{ij}=g_jB_{ji}$ to machine precision for the parsed model |
| C | **Detailed balance** | Force $J_\nu = B_\nu(T)$ at all depths → `solve_SEE` must return exactly the Saha–Boltzmann populations (rel. error $<10^{-10}$). *The single most powerful NLTE test.* A failure isolates the rates, the profile, the bf integrals or the Saha ratio |
| D | Collision-dominated limit | Multiply all $C$ by $10^6$ → populations → LTE, $S_\nu\to B_\nu$, emergent $I\to B(T(\tau_\nu=1))$ |
| E | Formal solver, linear source | $S=a+b\tau \Rightarrow I(0,\mu)=a+b\mu$ **exactly** (Eddington–Barbier is exact here). Sweep $\Delta\tau\in[10^{-10},10^{4}]$ to expose cancellation |
| F | Formal solver, constant source | $I = S(1-e^{-\tau}) + I_0e^{-\tau}$ |
| G | **Λ\* verification** | Perturb $S$ at depth $k$ by $\delta$, re-run the formal solution, measure $\partial J_k/\partial S_k$ numerically; must match $\Lambda^*_{kk}$ to $<1\%$ |
| H | Isothermal slab | Static, optically thick, isothermal → $J_\nu = B_\nu$ to machine precision. **Catches angular-quadrature normalization errors** |
| I | Two-level atom, constant properties | $S(0)/B \to \sqrt{\epsilon}$ at the surface; thermalization at $\tau\sim1/\epsilon$. Compare against the Avrett–Hummer solution |
| J | Λ\*-off equivalence | Setting $\Lambda^*\equiv0$ must converge to the **same** fixed point as the MALI run (slower). Different answers ⇒ broken preconditioning |
| K | Grid independence | Double $N_{\rm depth}$, $N_\nu$, and $N_{\rm rays}$ independently → emergent $I$ changes $<10^{-3}$ relative |
| L | Conservation | $\sum_i n_i = n_{\rm elem}$ at every depth, every iteration; all $n_i>0$ |
| M | LTE continuum | Lines off → continuum should match an LTE continuum reference to ~1% |

For each failing test, file a finding **before** attempting a fix.

---

## PHASE 4 — BENCHMARK AGAINST SNAPI

Working directory: `/home/andreuva/Documents/firtez-nlte/tests/compare_snapi`

1. Inventory what is there: which atmosphere (FAL-C? VAL3C? a MURaM/Bifrost column?), which
   atoms, which reference outputs, and the units and grids of the SNAPI results. Do not assume —
   read the files.
2. Establish that the two codes are being asked the **same question** before comparing answers:
   identical atmosphere interpolated onto identical depth points, identical abundances, identical
   atomic model and collisional data, same treatment of $n_e$, same background opacity sources,
   same CRD/PRD choice, same microturbulence. **List every remaining difference explicitly.**
   Most "benchmark disagreements" are setup mismatches, and diagnosing physics against a
   mismatched setup will send you chasing ghosts.
3. Compare, in this order (each one localizes a different layer of the problem):
   - **LTE continuum** at several wavelengths → tests background opacity, EOS, $n_e$.
   - **Departure coefficients** $b_i(z)$ per level → tests the rates and SEE, independent of the
     formal solver.
   - $\bar{J}$ and the line source function vs height → tests the radiative transfer.
   - **Emergent intensity** profiles: line-core intensity, core wavelength, wing shape,
     equivalent width, and the height of $\tau_\nu=1$ across the profile.
4. Metrics: relative RMS in the core (±3 Doppler widths) and in the wings, separately;
   max relative deviation of $b_i$ per level; $\Delta$(line-core intensity) in %.
5. **Interpret the residuals physically.** A core-only discrepancy points at $\bar{J}$,
   $\Lambda^*$, or the profile; a wing-only discrepancy points at broadening or PRD; a uniform
   offset points at units, abundances, or the boundary condition; a $\mu$-dependent offset points
   at the lower boundary or the angular quadrature.

---

## PHASE 5 — DELIVERABLES

1. **`audit/FINDINGS.md`** — the register, sorted by severity, each entry in this format:

   ```
   ## F-012 — [CRITICAL|HIGH|MEDIUM|LOW] — <one-line title>
   Class:      A (wrong physics) | B (wrong numerics) | C (non-standard)
   Location:   formal_solver.py:88-104
   Standard:   HM2014 eq. 12.xx / RH `piecewise_linear_1D` / Lightweaver `<fn>`
   Correct:    <equation>
   Implemented:<equation as coded>
   Impact:     <what observable is wrong, by how much, where in the atmosphere>
   Evidence:   <test that fails, or numerical demonstration>
   Fix:        <minimal patch>
   Verified:   <test that now passes; effect on the SNAPI comparison>
   ```

2. **Patches** — one commit per finding, message referencing the finding ID.
3. **`tests/test_physics.py`** — the Phase 3 suite, all passing.
4. **`audit/BENCHMARK.md`** — SNAPI comparison with plots and quantitative metrics, before and
   after the fixes.
5. **`audit/OPEN_QUESTIONS.md`** — everything you could not resolve from the code alone, and what
   would resolve it.

## PRIORITIZATION

Work in this order, because errors in the earlier items masquerade as errors in the later ones:

1. Units and conventions (Phase 1)
2. Formal solver correctness + boundary conditions + angular quadrature (2.7, 2.8, 2.9)
3. Profile normalization and the frequency grid (2.3)
4. Detailed balance / LTE recovery (Test C, Test D)
5. Λ\* consistency and MALI preconditioning (2.12, Tests G and J)
6. Background opacity and scattering (2.10)
7. Broadening recipes (2.13)
8. Everything else

## SEED LEADS

These are places I already consider suspicious from a read of `main.py` alone. **Verify each
independently — do not assume any is a real bug, and do not assume this list is complete.**

1. Voigt profile truncated by `max_delta_nu` then numerically renormalized (~L287–290). The
   author's own TODO says it "matters so much" — that sensitivity is itself the diagnostic.
2. Numerical rather than analytic profile normalization, which hides frequency-grid errors.
3. `0.5 * weigths[ir]` for $J$ and $\bar{J}$ — verify the quadrature weight convention.
4. Lower boundary $I = B(T_{\rm bottom})$ instead of the diffusion approximation (L218, L474).
5. $\Lambda^*$ forced to zero at the boundary node (L231) while the rates are preconditioned
   with it.
6. No line-of-sight velocity in `dop_freq` (L260) — no $\mu$ dependence in the profile.
7. `lte_ratios_photoionization` computed once outside the loop (L85–86) — correct only if it is
   a pure function of $T$ and $n_e$.
8. `hnu_grid[hnu_grid == 0] = 1e-100` (L71) — why is there a zero frequency?
9. `if ray == np.max(rays)` (L309) selects the emergent ray by $\mu$ value, not $\mu=1$.
10. The "10⁻⁴ scalar" unit comment in the linear Stark term (L189–190).
11. `eval(configuration["debug"])` (L40) vs `configuration.get("debug", False)` (L534).
12. `tau_depth[0,:]` set to a nonzero value under a comment saying it is zero (L476).

## FIRST ACTION

Do **not** start editing. Begin with Phase 0 and report the inventory and the units register.
Then propose your ordered attack plan and wait for confirmation before making any change to the
physics.
