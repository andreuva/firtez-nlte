# Open questions — what I could not settle from the code alone

Each entry states what is uncertain, why the code cannot settle it, and **what specific
information would settle it**. Nothing here is a guess presented as a conclusion.

---

## Q-01 — **RESOLVED** — SNAPI did not use PRD; neither does Lightweaver

The author confirmed **SNAPI ran CRD**, and Lightweaver's H and Ca II lines are all
`VoigtLine` (CRD) too. All three codes therefore share the same redistribution treatment,
and F-012 (this code being pure CRD) is **not** a source of benchmark disagreement.

An earlier revision of this file promoted this question to "decisive" on the grounds that the
Ca II 3d photoionization edges (121.750/121.840 nm) lie inside the Ly-alpha window
(119.93-123.26 nm), and that our Ca II 3d photoionization rates were 2.3-3.4x Lightweaver's
while the ground state, whose edge lies outside, agreed to 0.982. The overlap and that
pattern are both real, but the cause was **F-017** — bound-free opacity of active atoms
counted twice — not redistribution. Recorded because the reasoning looked compelling and was
wrong: a suggestive correlation is not a mechanism.

The residual F-012 caveat stands for *absolute* work on Ca II H & K and Ly-alpha, where PRD
matters physically. It just does not affect this benchmark, whose synthesised spectrum covers
only 853.7-854.7 nm.

## Q-02 / Q-08 — **RESOLVED** — the electron density agrees with SNAPI to 0.02%

*(Originally two open questions: whether SNAPI evolves n_e from the NLTE populations, and
whether this code's 92-element EOS is consistent with the FAL-C `pel` column. One
measurement settled both.)*

Comparing three electron densities on the 128-point FAL-C grid — SNAPI's converged n_e (the
13th column of `atm_lvl_pops`, last iteration), the FAL-C `pel` column via n_e = p_e/k_BT,
and this code's `compute_background_eos`:

| z (km) | T (K) | n_e (FAL-C p_el) | n_e (SNAPI) | n_e (our EOS) | ours/FAL-C | SNAPI/FAL-C |
|---|---|---|---|---|---|---|
| −90 | 9168 | 3.044e15 | 3.086e15 | 3.086e15 | 1.014 | 1.014 |
| 230 | 4922 | 3.018e12 | 2.613e12 | 2.515e12 | 0.833 | 0.866 |
| 550 | 4517 | 1.868e11 | 1.607e11 | 1.545e11 | 0.827 | 0.860 |
| 934 | 5859 | 1.273e11 | 3.406e11 | 3.408e11 | 2.676 | 2.675 |
| 1350 | 6766 | 1.122e11 | 5.113e11 | 5.113e11 | 4.556 | 4.557 |
| 1942 | 8164 | 7.669e10 | 1.820e11 | 1.820e11 | 2.373 | 2.373 |

**our EOS / SNAPI: median 1.0000, min 0.960, max 1.002.**

Two conclusions:

1. **SNAPI does not use the FAL-C `pel` column either** — both codes depart from it by the
   same large factor (median 2.48, up to 4.69 in the chromosphere). So this code's decision
   to parse `pel` and then ignore it, solving n_e from a 92-element LTE equilibrium on
   (T, p_g), **matches SNAPI's convention**. That was the single largest setup risk and it
   is now closed.
2. **`compute_background_eos` is validated against an independent implementation** to 0.02%
   in the median. The worst point is 4%.

The remaining n_e question is narrower and is folded into Q-06: the NLTE feedback loop
(`atoms.py:621-643`) moves n_e away from this validated baseline during the iteration, while
`bg_species` stays frozen at it.

## Q-03 — Which `q_wing` / `N_lambda` did SNAPI use, and is the comparison grid-converged?

**Why it matters.** F-003: the line grids are built on a fixed characteristic Doppler width,
and the residual profile-normalisation error after F-002 is ordinary trapezoid error that
depends entirely on `N_lambda` (measured 1.102 → 1.022 → 1.005 → 1.001 under successive
doubling for H 1→5). SNAPI's `debug_norm` array has a maximum of **1.0145** — strikingly
close to this code's Ca II value of 1.014 — which suggests SNAPI uses a comparable grid and
also renormalises. That is suggestive, not proof.

**What would settle it:** SNAPI's per-transition wavelength grid definition (the `NLAMBDA`,
`QCORE`, `QWING` entries of its atom file), plus confirmation that `debug_norm` is
∫φdλ before renormalisation.

---

## Q-04 — Is `n_gaus` matched between the two codes?

**Why it matters.** F-014 established that even with the corrected hemispheric quadrature,
Ca II J̄ moves **1.7% between 6 and 10 rays**, and that the repository default was
`n_gaus: 5`. A 1.7% J̄ difference is comparable to the population differences the benchmark
is trying to resolve. I raised the comparison config to `n_gaus: 12`, which is converged to
~1e-3 in the median, but that is my choice, not a match to SNAPI.

**What would settle it:** SNAPI's angular quadrature setting (number of rays per hemisphere
and the rule used).

---

## Q-05 — What is the correct Γ_rad for the hydrogen subordinate lines?

F-010 established that the config's Γ_rad equals Σ_k A_jk exactly and omits Σ_k A_ik, making
it 3.6×–42× too small for Hα, Hβ, Hγ, Paα and Brα. **Ca II is unaffected**, so the benchmark
is safe. But I deliberately did not edit atomic data.

**What would settle it:** confirmation from you that the atom file is meant to carry the
full Γ_rad (in which case the values should be regenerated as Σ_k A_jk + Σ_k A_ik), or that
the code is meant to add the lower-level contribution itself (in which case it is a code
change, and RH/Lightweaver's convention should be checked — I did not verify which
convention the reference atom files use).

---

## Q-06 — Should the frozen background species track the evolving n_e?

F-006 fixed the bound-free Saha ratio, which is exactly linear in n_e and so could be
rescaled in one line. But `atmosphere.bg_species` — the H⁻, H I, H II, He and metal densities
that feed every background opacity — is computed **once** at `ne_bg` (`atmosphere.py:84`) and
never updated. H⁻ opacity is proportional to n_e, and the H⁻ *density* itself depends on n_e,
so the background continuum carries the same staleness.

**Why I did not fix it:** unlike the Saha ratio, this is not a simple rescale —
`compute_background_species` would have to be re-run each iteration (a full 92-element
equilibrium per depth), which is a performance and design decision, not a one-line
correction. It also may be deliberate: freezing the background is a common and defensible
choice (RH's `background()` is recomputed only when `solve_ne` is active).

**What would settle it:** your intent for the background — frozen LTE background (document
it) or self-consistent (needs the re-evaluation, and a decision on cost).

---

## Q-07 — **RESOLVED** — the `max(J_eff, 0)` clamp never fires

`atoms.py:775` clamps J̄_eff = J̄ − Λ*S_old at zero. Unlike the Λ̄* clip (which is provably
inert once the profile is normalised), **this clamp changes the converged solution when it
fires**. I proved algebraically (test `test_J2_lambda_star_cancels_in_the_net_rate`) that the
preconditioning is exact *only while J_eff > 0*.

Before F-002 it certainly fired, because Λ̄* was corrupted. Whether it still fires in a
converged production run I could not confirm — it needs instrumentation inside a full run.

**ANSWER: it fires 0 of 1280 (depth, line) pairs** in the converged solution, including at the
pathological top boundary point. The clamp is inert and does not move the fixed point. It
should be replaced by an assertion so that any future activation is loud rather than silent.
See `FINDINGS.md` F-016 step 5.

---

## Q-09 — Why does the 95 nm Lyman line's J̄ not converge under ray refinement?

`test_K_angular_quadrature_converges` documents that H line 3 (94.98 nm) at the top of the
82-point model (T = 1e5 K, J̄ ≈ 1e-8) does not converge with ray count at all — the change
stays at ~5e-2 from 16 to 128 rays, while the median converges cleanly to 3e-7.

My working explanation is F-003: at 1e5 K the real Doppler width is 13.5× the grid's
characteristic width, so the line's window spans only ±2.2 real Doppler widths and the
profile is truncated inside its own core. That would make the integrand grid-limited rather
than quadrature-limited. **I did not confirm this**, and it does not affect the FAL-C
benchmark (which tops out at 9168 K).

**What would settle it:** re-running that convergence study with `q_wing` raised for the
Lyman lines. If the non-convergence disappears, F-003 is confirmed as the cause.

---

## Q-10 — Are the tests meant to be version-controlled?

`.gitignore` line 221 ignores `tests*`, and no test file in the repository is tracked. That
is reasonable for the ~30 MB of SNAPI benchmark data under `tests/compare_snapi/`, but it
also excluded the suite itself, which conflicts with the requirement that tests be
reproducible from a clean checkout.

I force-added **only** `tests/test_physics.py` and left the data ignored. If you would rather
the pattern were narrowed (e.g. `tests/**/*.npz`, `tests/**/*.dat`, `tests/**/*.png`), that
is a one-line change to `.gitignore` — but it is your repository convention to set, not mine.

---

## Q-11 — **RESOLVED** — negative total opacity at the top boundary (see F-016)

The LTE audit found **zero** negative opacity samples over the whole depth × frequency grid
(refuted lead R-15). The converged NLTE run tells a different story: `main.py` printed
**"Negative absorption at depth 127"** on essentially every iteration (25 of the first 26),
always at depth 127 — the topmost point, z = 1942 km — together with 288 negative
source-function warnings.

**Why this matters.** `formal_solver.py:920-921` computes `S = emis/abs` with **no guard**,
and `formal_solution` forms Δτ = ½(χ_M+χ_O)|dz/μ|. A negative χ gives a negative Δτ, hence
`exp(-Δτ) > 1` — an *amplifying* step rather than an attenuating one — and a negative or
divergent source function. Depth 127 is the last point of every upward ray, so this enters
the emergent intensity directly. The code only prints a warning; it does not correct or
clamp (the clamps are commented out at `formal_solver.py:390,393`).

**What I could not determine:** the magnitude (is |χ| tiny, making this cosmetic, or is it
comparable to the background?) and the cause. Two candidates, distinguishable by inspection:
- a genuine population inversion in one of the Ca II or H lines (n_u B_ul > n_l B_lu), which
  is physical at the top of a chromosphere but should normally be swamped by the background;
- the bound-free term `sigma_nu * (n_l - n_u (n_l*/n_u*) exp(-h nu/kT))`
  (`formal_solver.py:379-380`) going negative under strong overionization, i.e. stimulated
  recombination exceeding photoionization absorption.

**What would settle it:** print χ_line, χ_bf and χ_background separately at depth 127 for
the offending frequencies in a converged run. That localises it to a transition in one line.
This is the first thing I would do with more budget — it is the only defect this audit
surfaced that is **not** understood.

---

## Q-12 — **CLOSED** — the 4p3/2 rate-matrix oddity was not the cause of anything

SNAPI's stored rate matrix gives levels 3 and 4 bound-free rates whose ratio is 17-40 where
their byte-identical cross-section table demands ~1.90 by the Milne relation. That
observation stands. It was originally filed as F-015, the supposed cause of the chromospheric
Ca II overionization.

**It was not.** Lightweaver agreed with SNAPI's populations and disagreed with ours, showing
the dominant error was ours (F-017). F-015 is withdrawn. The oddity in the *serialised*
matrix may simply reflect how SNAPI writes that array rather than the rates it used; nothing
in the benchmark depends on resolving it, and it is recorded here only so it is not
rediscovered and misread the same way.

**Process note kept deliberately:** the mistake was attributing a defect to the reference
data from an internal-consistency argument, without an independent third implementation.
Lightweaver was installed in the project environment throughout and settled it in one run.
