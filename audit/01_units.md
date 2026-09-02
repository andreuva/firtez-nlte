# Phase 1 — Units & Conventions Register (read-only)

Everything below states the units the code *implies by use*, checked against the expected CGS
convention. ✔ = verified consistent; ⚠ = flagged for the Phase 2 equation audit.

## 1. Core symbols

| Symbol | Code variable | Expected (CGS) | Implied by use | Status |
|---|---|---|---|---|
| ν | `frequency_grid` | Hz | Hz (from c/λ, λ in cm) | ✔ |
| dν weights | `weigths_freq_grid` | Hz | Hz, global trapezoid (atoms.py:538-541) | ⚠ global, not per-line (→2.3) |
| I_ν, J_ν, B_ν | `I_o`, `J_grid`, `plank()` | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ | same everywhere (per-frequency; no per-λ leakage found in the solver; per-λ conversions only in plotting/benchmark scripts) | ✔ |
| S_ν | `emis/abs` | erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ | η/χ | ✔ |
| χ_ν | `abs` | cm⁻¹ | cm⁻¹ | ✔ |
| η_ν | `emis` | erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹ | ✔ |
| E (levels) | `level.energy` | erg | config cm⁻¹ ×hc (atoms.py:332) | ✔ |
| E_Ryd | `E_Ryd_erg` = 2.179872e-11 | erg (∞-mass) | reduced-mass corrected where needed (main.py:169, atoms.py:100) | ✔ |
| χ_ion thresholds | `XI`, `XII` (chemeq) | eV, converted via `CSAHA2=eV/kB` → K | ✔ (H row = H⁻/H convention, see inventory §5.7) | ✔/⚠ |
| n_e, n_H, n_i | `ne`, `nh`, `populations` | cm⁻³ | cm⁻³ | ✔ |
| abundance | `atom.abundance` | number ratio vs H (linear) | H=1.0, Ca=2.138e-6 | ✔ (value vs SNAPI ⚠) |
| `he_abund` | N_He/N_H | 10^(11−12)=0.1, **number ratio, not mass fraction** | used as multiplier on nHGround in vdW He term (main.py:130,148) — Lightweaver does the same | ✔ (approximation: n_He ≈ he_abund·n_H(ground), class C) |
| ξ (microturb) | `turbulent_velocity` | cm s⁻¹, a velocity (not FWHM) | 3e5 cm/s; enters √(2kT/m+ξ²) | ✔ |
| Δν_D | `doppler_widths` | Hz | ν₀/c·√(2kT/m+ξ²) — the `2` is the most-probable speed ✔ (atoms.py:408) | ✔ |
| z | `zgrid` | cm, increasing upward | config km ×1e5 | ✔ |
| p_g, p_e | `pg`, `pel` | dyn cm⁻² | pel parsed but unused | ⚠ (silent) |
| T | `temp` | K | ✔ |
| σ (cross-sections) | `photoionization_alphas`, bf tables | cm² | config in **m²**, ×1e4 at parse (atoms.py:193,240) | ✔ |
| Barklem σ | ABO tables | atomic units a₀² | ×a0² at atoms.py:120 | ✔ |
| A_ul | `line.Aul` | s⁻¹ | verified vs NIST | ✔ |
| B_ul, B_lu | `line.Bul/Blu` | (erg s⁻¹cm⁻²Hz⁻¹sr⁻¹)⁻¹ s⁻¹ (J̄-convention, not u_ν-convention) | A/(2hν³/c²) | ✔ |
| φ_ν | `voigt_norm` | Hz⁻¹ | after numerical renorm vs Hz weights | ✔ units / ⚠ normalization source (→2.3) |
| C_ij | `C_matrix` | s⁻¹ | per-particle rates ✔ | ✔ |
| R_ik, R_ki | `photoionization_rates` etc. | s⁻¹ | 4π∫(α/hν)J dν ✔ | ✔ |
| τ | `delta_tauMO` | dimensionless | 0.5(χ_M+χ_O)|dz/μ| | ✔ |

## 2. Every explicit π / 2 / 4π factor and its origin

| Location | Factor | Origin | Status |
|---|---|---|---|
| atoms.py:372 `Aul = 8π²e²ν²/(c³mₑ)(g_l/g_u)f` | 8π² | classical oscillator: A = (2πe²ν²/mₑc³)·(8π²/…) — net formula equals the standard γ_cl·3·(g_l/g_u)f relation; **verified numerically vs NIST** | ✔ |
| atoms.py:374 `Bul = c²/(2hν³)·Aul` | 2 | Planck 2hν³/c² | ✔ |
| main.py:296, formal_solver.py:361-362 `hν₀/4π` | 4π | per-steradian line opacity/emissivity with ∫φdν=1 | ✔ |
| main.py:282, formal_solver.py:346 `a = Γ_tot/(4π Δν_D)` | 4π | Γ [s⁻¹] is the *full* damping rate (sum of FWHM-type rates); Lorentzian HWHM in ν = Γ/4π | ✔ convention; ⚠ each Γ contribution must itself be a full width (→2.13) |
| main.py:303/305/307 `0.5·w_ir·(...)` | 1/2 | J = ½∫₋₁¹ I dμ with GL weights summing to 2 | ✔ (verified Σw=2) |
| main.py:330-331 `4π·Σ w α/hν (…)` | 4π | R_ik = 4π∫(α_ν/hν)J_ν dν (HM2014 eq. 9.44-type with J from I/4π) | ✔ |
| main.py:124/144 `C6 = 2.5 q²α_H·2π(Z a₀)²/h·ΔR̄²` | 2π | Unsold C6 (HM2014 §8.3; matches Lightweaver `c6_barklem`-free VdwUnsold) | ⚠ verify numeric vs LW (2.13) |
| main.py:127-128 `(8kB/(π m)(1+m/m_H))^0.3` | π | v̄=√(8kT/πμ); exponent 0.3 = (v²)^0.3 → v^0.6 | ✔ algebra checked |
| main.py:163 `C_stark=8kB/(π m)` | π | same v̄ for quadratic Stark, ^(1/6) | ✔ |
| main.py:176 `C4 ∝ 2π a₀²/h /(18 Z⁴)` | 2π, 18 | Traving 1960 C4 as in RH/LW | ⚠ verify numeric (2.13) |
| main.py:179 `11.37 (coeff·C4)^(2/3)` | 11.37 | Lindholm quadratic-Stark coefficient | ✔ standard |
| main.py:190 `cc·4π·0.425` | 4π·0.425 | Sutton (1978) linear Stark, LW form; the dropped 1e-4 must equal (1e6 cm⁻³/m⁻³)^(2/3)=1e4 inverse | ⚠ explicit dimensional check in 2.13 |
| atoms.py:122 `2·(4/π)^(α/2)Γ(2−α/2)·v̄·σ(v̄/v₀)^(−α)` | 2, 4/π | ABO full width (factor 2 = HWHM→FWHM as in LW `Broadening.py`) | ⚠ verify factor-2 & T-exponent vs LW (2.13) |
| atoms.py:715 `C0 = 8.629e-6` | — | (2πmₑ)^(−1/2)(kB)^(−1/2)·2πa₀²·√(2E_Ryd/mₑ)... standard Υ prefactor cm³ s⁻¹ K^(1/2) | ✔ value |
| atoms.py:721/726 `×1e6` | 1e6 | RH tabulates CE/CI in m³; →cm³ | ✔ intent; ⚠ verify against RH H.atom values (2.6) |
| atmosphere.py:109 `saha_const=(2πmₑkB/h²)^1.5`, `Phi=(2/ne)·saha·T^1.5` | 2π, 2 | Saha with electron-spin 2 | ✔ |
| chemeq.py:429 `CSAHA1=(h²/2πmₑkB)^1.5` | 2π | inverse Saha | ✔ |
| formal_solver.py:718 (dead code) `σ_T=(8π/3)(e²/mₑc²)²` | 8π/3 | Thomson | ✔ (dead); live value 0.6653e-24 cm² at L446 ✔ |
| formal_solver.py:914 `Δτ=0.5(χ_M+χ_O)|dz/μ|` | 1/2 | trapezoidal χ mean | ✔ convention (order → 2.7) |

## 3. Conventions that differ from RH/Lightweaver/SNAPI (candidate class C or B)

1. **Lower boundary** I⁺=B_ν(T_bottom) (main.py:218,474) instead of the diffusion approximation
   B+μ dB/dτ. (→2.8)
2. **Λ\*** at the boundary node forced to 0 (main.py:231) — consistent with a *prescribed* I⁺,
   but the prescription itself is the non-standard part. (→2.8)
3. **CRD everywhere** including Lyα/Lyβ and Ca II H&K which are flagged "PRD" in the config
   and are PRD in RH/LW/SNAPI-with-PRD. (→2.14; SNAPI run settings must be checked in Phase 4.)
4. **Global frequency-grid quadrature weights reused for every purpose**: line J̄, profile
   normalization, and bf integrals all use the same global trapezoid weights; RH/LW use
   per-transition subgrids/weights. (→2.3, 2.5)
5. **Voigt truncation at `max_delta_nu` + numerical renormalization** (main.py:287-290) —
   no analog in RH/LW, which evaluate each line only on its own wavelength window. (→2.3)
6. **Scattering emissivity** uses previous-iteration J (σ·J_nu at formal_solver.py:455) —
   iterated, not frozen ✔, but Thomson+Rayleigh σ·J with J updated only once per Λ-iteration;
   also the `bg_species` and H⁻ densities stay frozen at ne_bg while `atmosphere.ne` evolves. (→2.10, 2.11)
7. **ne update** by Δcharge with 0.5 damping and a 1e-6·nh metal floor (atoms.py:629-640) —
   RH solves charge conservation self-consistently; SNAPI differs again. The floor is a clamp
   masking the wrong LTE H II baseline (inventory §5.7). (→2.11)
8. **Emergent intensity during iterations** taken at the largest quadrature μ (0.932), not μ=1
   (main.py:309) — diagnostics only; final spectrum is a genuine μ=1 pass. Cosmetic.
9. **Wavelength-symmetric line grids** (atoms.py:489): symmetric in λ, slightly asymmetric in ν —
   matches Lightweaver behaviour. ✔ class C consistent with LW.
10. **Population clamp** `populations_new[populations_new<0]=1e-100` (atoms.py:815) and
    `L_star` clip to [0, 0.9999999] (atoms.py:760) and `J_eff = max(J̄−Λ*S, 0)` (atoms.py:775):
    three clamps inside the SEE that can move the converged fixed point. (→2.12, Test J)

## 4. Existing clip/floor inventory (rule 5: each is a potential mask)

| Location | Clip | Why it can trigger | Verdict pending |
|---|---|---|---|
| main.py:71 | hnu==0 → 1e-100 | never (no zero freq) | dead code |
| main.py:86 | max(n_u*,1e-100) | LTE upper-stage pop can underflow at cold depths? H II pop is wrong anyway (F-001) | Phase 2.5 |
| main.py:298 | abs_O>1e-100 mask for opacity_ratio | avoids 0/0 in gaps where line+bg opacity ~0 | Phase 2.12 |
| atoms.py:727/736 | max(lte_pop,1e-100) | same as above | Phase 2.6 |
| atoms.py:760 | clip(Λ̄*,0,0.9999999) | quadrature overshoot of Λ̄*>1 — symptom of the §2.3 weight problem | Phase 2.12 |
| atoms.py:775 | max(J_eff,0) | J̄<Λ̄*S_old happens when Λ̄* overshoots | Phase 2.12 |
| atoms.py:815 | negative pops → 1e-100 | SEE matrix conditioning / bad rates | Phase 2.1 |
| atoms.py:630-633 | ne floor 1e-6·nh | masks wrong Δcharge baseline from F-001 | Phase 2.11 |
| formal_solver.py:914 | +vacuum_CGS (1e-200) in Δτ | avoids 0/0 for S=η/χ in true vacuum | benign but S=η/χ with χ→1e-200 explodes if η>0 — Phase 2.7 |
| formal_solver.py:920-921 | S=emis/abs unguarded | division by ~0 in gaps → inf/NaN risk | Phase 2.7 |
| psi_lin small branch ≤0.1 | Taylor series | verified accurate | ✔ |
| plank OVERFLOW_LIMIT 200 | Wien switch | verified stable | ✔ |
| _opac_* np.maximum(...,0) | Wittmann ports | negative table extrapolations | Phase 2.10 |

## 5. Per-frequency vs per-wavelength cross-checks for Phase 4

- FIRTEZ-py: I_ν per Hz. SNAPI debug J: per cm (wavelength). SNAPI synth spectrum: per cm.
  Conversion I_λ = I_ν·c/λ² (already used in `run_full_comparison.py:163`). Any comparison
  must also match μ (SNAPI synth = μ=1 disk centre? verify), and SNAPI's J̄ is per-transition
  profile-weighted — same object as `atom.Js` only if SNAPI's norm convention matches.
