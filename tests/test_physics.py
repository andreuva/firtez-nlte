"""
Phase-3 analytic regression suite for the firtez-nlte solver.

Every test here has an answer known in closed form, so a failure localises the bug rather
than merely signalling that "something moved". Run from the repository root:

    python -m pytest tests/test_physics.py -v

Deterministic by construction: no random numbers are used anywhere. Where a seed would be
needed it is set explicitly (see `_rng`).

Test letters refer to the audit brief's Phase-3 table; findings to audit/FINDINGS.md.
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from constants import (kB_CGS, h_CGS, c_CGS, m_e_CGS, m_u_CGS)          # noqa: E402
from atmosphere import (Atmosphere, compute_lte_populations,             # noqa: E402
                        get_angular_quadrature_1D)
from atoms import (MultiLevelAtom, create_frequency_grid, solve_SEE, solve_atom,  # noqa: E402
                   _hydrogen_collider_densities,
                   compute_line_frequency_weights, continuum_frequency_weights,
                   init_line_broadening)
from formal_solver import (plank, voigt, formal_solution, psi_lin,       # noqa: E402
                           get_RT_coefficients, line_profile, line_damping,
                           _seaton)

CONFIG = os.path.join(REPO_ROOT, "config_H_Ca_Mg_Na.json")

# Deterministic generator, used only to pick perturbation directions in Test G.
_rng = np.random.default_rng(20240501)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def model():
    """Fully initialised atoms + atmosphere + frequency grid, exactly as main.py builds them."""
    warnings.filterwarnings("ignore")
    with open(CONFIG) as f:
        cfg = json.load(f)
    atmosphere = Atmosphere.from_dict(cfg["atmosphere"])
    atoms = [MultiLevelAtom.from_dict(a) for a in cfg["atoms"]]
    v_turb = cfg["atmosphere"]["turbulent_velocity"]
    for atom in atoms:
        atom.populations = compute_lte_populations(atom, atmosphere)
        atom.lte_populations = atom.populations.copy()
        atom.compute_doppler_widths(atmosphere, v_turb)
    freq, weights = create_frequency_grid(atoms, atmosphere=atmosphere, v_turb=v_turb)
    compute_line_frequency_weights(atoms, freq)
    init_line_broadening(atoms, atmosphere)

    wl_nm = (c_CGS / freq) * 1e7
    for atom in atoms:
        atom.photoionization_alphas = np.zeros((len(atom.continua), len(freq)))
        atom.lte_ratios_photoionization = np.zeros((atmosphere.Ndepth, len(atom.continua)))
        for ic, cont in enumerate(atom.continua):
            atom.lte_ratios_photoionization[:, ic] = (
                atom.lte_populations[:, cont.lower_level_index]
                / np.maximum(atom.lte_populations[:, cont.upper_level_index], 1e-100))
            atom.photoionization_alphas[ic, :] = cont.alpha(wl_nm, atom.levels)

    atmosphere.J_nu = np.zeros((atmosphere.Ndepth, len(freq)))
    for iz in range(atmosphere.Ndepth):
        atmosphere.J_nu[iz, :] = plank(freq, atmosphere.temp[iz])

    return dict(atoms=atoms, atmosphere=atmosphere, freq=freq, weights=weights,
                v_turb=v_turb, cfg=cfg)


def _depths(atmosphere):
    """Bottom, middle and top of the model."""
    return [0, atmosphere.Ndepth // 2, atmosphere.Ndepth - 1]


def _march(mu, tau_grid, S_of_tau, I_bottom):
    """
    March the formal solver upward through a 1-D slab on a *vertical* optical-depth grid.

    tau_grid runs from the deepest point to the surface. dz is passed as the vertical
    dtau with chi = 1, so formal_solution forms dtau_path = dtau_vertical / |mu|.
    """
    I = np.array([I_bottom])
    for k in range(1, len(tau_grid)):
        dtau_v = abs(tau_grid[k - 1] - tau_grid[k])
        I, _ = formal_solution(mu, I, dtau_v,
                               np.array([S_of_tau(tau_grid[k - 1])]),
                               np.array([S_of_tau(tau_grid[k])]),
                               np.ones(1), np.ones(1))
    return I[0]


# ===========================================================================
# Test A — profile normalisation (audit F-002)
# ===========================================================================

def test_A_profile_normalisation_before_renormalisation(model):
    """
    int(phi dnu) must be 1 for the *analytic* profile, per line and per depth, using the
    quadrature the code actually integrates with -- BEFORE any numerical renormalisation.

    Regression guard for F-002: on the unpatched code this reached 13.31 for H Brackett-alpha
    at the top of the atmosphere, because the global trapezoid weight of the window-edge
    point spanned the empty gap to the next spectral block.

    The tolerance is 12%, not 1e-6, and that is deliberate: what remains is genuine
    second-order trapezoid error on the coarse exponentially-spaced line subgrids
    (doubling N_lambda gives 1.1020 -> 1.0217 -> 1.0049 -> 1.0012). That residual is what
    the renormalisation legitimately absorbs, exactly as Lightweaver's `wphi` does. The
    test exists to catch the *pathological* weights, which were an order of magnitude out.
    """
    atoms, atm, freq = model["atoms"], model["atmosphere"], model["freq"]
    from scipy.special import wofz
    worst, worst_id = 0.0, None
    for atom in atoms:
        for il, line in enumerate(atom.lines):
            gamma = sum(n.get("gamma", 0.0) for n in line.broadening.natural)
            for iz in _depths(atm):
                dnuD = atom.doppler_widths[iz, il]
                a = gamma / (4.0 * np.pi * dnuD)
                phi = wofz((freq - line.nu0) / dnuD + 1j * a).real / (np.sqrt(np.pi) * dnuD)
                integral = np.sum(phi[line.grid_mask] * line.freq_weights[line.grid_mask])
                if abs(integral - 1.0) > worst:
                    worst, worst_id = abs(integral - 1.0), f"{atom.name} line {il} iz {iz}"
    assert worst < 0.12, f"int(phi dnu) off by {worst:.4f} at {worst_id} (F-002 regression)"


def test_A2_renormalised_profile_integrates_to_one(model):
    """After renormalisation the profile the solver actually uses must integrate to 1 exactly."""
    atoms, atm, freq = model["atoms"], model["atmosphere"], model["freq"]
    nH_ground = atoms[0].populations[:, 0]
    for atom in atoms:
        for line in atom.lines:
            for iz in _depths(atm):
                phi = line_profile(line, atom, iz, freq, atm, nH_ground[iz])
                assert np.isclose(np.sum(phi * line.freq_weights), 1.0, rtol=1e-12)


def test_A3_line_weights_are_local_not_global(model):
    """
    The per-line weight of any point must never exceed the global weight there, and the
    window-edge weight must be a genuine half-interval of the line's own grid.
    """
    atoms, freq, wq = model["atoms"], model["freq"], model["weights"]
    for atom in atoms:
        for line in atom.lines:
            idx = np.where(line.grid_mask)[0]
            w = line.freq_weights[idx]
            assert np.all(w > 0.0)
            assert np.all(w <= wq[idx] * (1.0 + 1e-12))
            nu = freq[idx]
            assert np.isclose(w[0], 0.5 * (nu[1] - nu[0]), rtol=1e-12)
            assert np.isclose(w[-1], 0.5 * (nu[-1] - nu[-2]), rtol=1e-12)


def test_A4_voigt_function_matches_faddeeva():
    """voigt() must return H(a,v) = Re[w(v+ia)], with correct Gaussian and Lorentz limits."""
    from scipy.special import wofz
    v = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
    for a in (1e-4, 1e-2, 0.1, 0.5, 1.0):
        ref = wofz(v + 1j * a).real
        assert np.allclose(voigt(v, a).real, ref, rtol=1e-4)
    assert np.isclose(voigt(np.array([2.0]), 0.0).real[0], np.exp(-4.0), rtol=1e-6)
    a = 0.1
    assert np.isclose(voigt(np.array([50.0]), a).real[0],
                      a / (np.sqrt(np.pi) * 2500.0), rtol=1e-2)


def test_A5_doppler_width_uses_most_probable_speed(model):
    """dnuD = nu0/c * sqrt(2kT/m + xi^2) -- the 2 is the most-probable speed, xi a velocity."""
    atoms, atm, v_turb = model["atoms"], model["atmosphere"], model["v_turb"]
    for atom in atoms:
        for il, line in enumerate(atom.lines):
            expected = (line.nu0 / c_CGS) * np.sqrt(
                2.0 * kB_CGS * atm.temp / (m_u_CGS * atom.mass) + v_turb**2)
            assert np.allclose(atom.doppler_widths[:, il], expected, rtol=1e-12)


# ===========================================================================
# Test B — Einstein relations
# ===========================================================================

def test_B_einstein_relations(model):
    """A_ji = (2 h nu^3/c^2) B_ji and g_i B_ij = g_j B_ji, to machine precision."""
    for atom in model["atoms"]:
        for line in atom.lines:
            gl = atom.levels[line.lower_level_index].g
            gu = atom.levels[line.upper_level_index].g
            assert np.isclose(line.Aul, (2 * h_CGS * line.nu0**3 / c_CGS**2) * line.Bul,
                              rtol=1e-14)
            assert np.isclose(gl * line.Blu, gu * line.Bul, rtol=1e-14)


def test_B2_f_to_A_conversion_against_nist(model):
    """Spot-check the f -> A conversion against NIST values (1% tolerance)."""
    expected = {(0, 1): 4.6986e8,    # H Lyman-alpha
                (1, 2): 4.4101e7,    # H-alpha
                (2, 3): 8.9860e6}    # H Paschen-alpha
    H = next(a for a in model["atoms"] if a.name == "H")
    for line in H.lines:
        key = (line.lower_level_index, line.upper_level_index)
        if key in expected:
            assert np.isclose(line.Aul, expected[key], rtol=1e-2), \
                f"A({key}) = {line.Aul:.4e}, NIST {expected[key]:.4e}"


# ===========================================================================
# Test C — detailed balance (the single most powerful NLTE test)
# ===========================================================================

def _build_lte_rates(model):
    """Populate Js / photoionization / recombination rates from J_nu = B_nu(T)."""
    atoms, atm, freq, wq = model["atoms"], model["atmosphere"], model["freq"], model["weights"]
    Nz = atm.Ndepth
    hnu, hnu3 = h_CGS * freq, h_CGS * freq**3
    J = np.zeros((Nz, len(freq)))
    for iz in range(Nz):
        J[iz, :] = plank(freq, atm.temp[iz])
    atm.J_nu = J
    nH_ground = atoms[0].populations[:, 0]
    for atom in atoms:
        atom.Js = np.zeros((Nz, len(atom.lines)))
        atom.Lambda_star_bar = np.zeros((Nz, len(atom.lines)))
        atom.photoionization_rates = np.zeros((Nz, len(atom.continua)))
        atom.recombination_rates = np.zeros((Nz, len(atom.continua)))
    for iz in range(Nz):
        for atom in atoms:
            for il, line in enumerate(atom.lines):
                phi = line_profile(line, atom, iz, freq, atm, nH_ground[iz])
                atom.Js[iz, il] = np.sum(line.freq_weights * J[iz, :] * phi)
            for ic, cont in enumerate(atom.continua):
                al = atom.photoionization_alphas[ic, :]
                act, cont_weights = continuum_frequency_weights(al, freq, wq)
                if act.size == 0:
                    continue
                stim = ((2.0 * hnu3[act] / c_CGS**2 + J[iz, act])
                        * np.exp(-hnu[act] / (kB_CGS * atm.temp[iz])))
                atom.photoionization_rates[iz, ic] = 4.0 * np.pi * np.sum(
                    cont_weights * (al[act] / hnu[act]) * J[iz, act])
                atom.recombination_rates[iz, ic] = 4.0 * np.pi * np.sum(
                    cont_weights * (al[act] / hnu[act])
                    * atom.lte_ratios_photoionization[iz, ic] * stim)
    return J


def test_C0_continuum_endpoint_weights_do_not_borrow_an_unrelated_interval():
    """A constant integrand on [1, 2] must not acquire half the interval to 100."""
    frequency = np.array([1.0, 2.0, 100.0])
    global_weights = np.array([0.5, 49.5, 49.0])
    alphas = np.array([1.0, 1.0, 0.0])

    idx, weights = continuum_frequency_weights(alphas, frequency, global_weights)

    assert np.array_equal(idx, np.array([0, 1]))
    assert np.array_equal(weights, np.array([0.5, 0.5]))
    assert np.sum(weights) == 1.0


def test_C0_every_continuum_threshold_is_an_exact_frequency_grid_point(model):
    atoms, freq = model["atoms"], model["freq"]
    for atom in atoms:
        for ic, cont in enumerate(atom.continua):
            edge_frequency = c_CGS / (cont.get_lambda_edge_nm(atom.levels) * 1e-7)
            assert np.any(freq == edge_frequency), (
                f"{atom.name} continuum {ic} threshold missing from global grid")


def test_C0_round_trip_keeps_nonzero_threshold_cross_section(model):
    atoms, freq = model["atoms"], model["freq"]
    for atom in atoms:
        for ic, cont in enumerate(atom.continua):
            edge_frequency = c_CGS / (cont.get_lambda_edge_nm(atom.levels) * 1e-7)
            idx = np.flatnonzero(freq == edge_frequency)[0]
            round_trip_wavelength = np.array([(c_CGS / freq[idx]) * 1e7])
            assert cont.alpha(round_trip_wavelength, atom.levels)[0] > 0.0, (
                f"{atom.name} continuum {ic} lost its threshold cross-section")


def test_C_detailed_balance(model):
    """
    With J_nu = B_nu forced at every depth, solve_SEE must return the Saha-Boltzmann
    populations. This exercises the rates, the profile, the bf integrals and the Saha
    ratio simultaneously. The conservation total is the sum of the explicit LTE levels,
    so this must hold even where a truncated atom represents only a small fraction of the
    element. On the old normalization Mg reached a common b=188.672 at 100,000 K.
    """
    atoms, atm = model["atoms"], model["atmosphere"]
    _build_lte_rates(model)
    ne_before = atm.ne.copy()
    reference = {a.name: a.lte_populations.copy() for a in atoms}

    for electron_mode in ("eos", "delta", "nlte"):
        atm.ne[:] = ne_before
        for atom in atoms:
            atom.populations = reference[atom.name].copy()

        solve_SEE(atoms, atm, electron_mode=electron_mode)

        assert np.max(np.abs(atm.ne / ne_before - 1.0)) < 1e-3, \
            (f"n_e drifted in {electron_mode!r} mode while the radiation field was "
             "already in detailed balance")

        for atom in atoms:
            ref = reference[atom.name]
            significant = ref / ref.sum(axis=1, keepdims=True) > 1e-8
            departure = atom.populations / np.maximum(ref, 1e-300)
            dev = np.max(np.abs(departure[significant] - 1.0))
            assert dev < 5e-3, \
                (f"{atom.name}: departure coefficients differ from one by {dev:.3e} "
                 f"under J=B in {electron_mode!r} mode")


def test_C2_lte_populations_conserve_particle_number(model):
    """
    sum_i n_i^LTE vs A_elem * n_H. Regression guard for F-001, where hydrogen summed to
    2.000 * N_total at cool depths and 0.0040 * N_total at 1e5 K.

    Two distinct things are checked, because they mean different things:

      - OVER-counting (ratio > 1) is always a bug: the partition function used for a stage
        is smaller than the sum of g over the levels actually modelled. That is precisely
        the F-001 signature and must never occur.

      - UNDER-counting (ratio < 1) is model incompleteness, not a code defect: the partition
        function legitimately accounts for levels the model atom does not carry. Mg I at
        1e5 K is the extreme case here (U(Mg II) = 377, one modelled Mg II level). What is
        asserted instead is that each model accounts for the whole element SOMEWHERE; if it
        never does, its level set or its stage labelling is wrong. Measured best/worst
        completeness for this configuration: H 1.00002/1.00000, Ca II 1.00006/0.99929,
        Mg I 0.99990/0.00530, Na I 1.00000/0.99867. Runtime warning in main.py flags the
        depths where it matters.
    """
    atoms, atm = model["atoms"], model["atmosphere"]
    for atom in atoms:
        ratio = compute_lte_populations(atom, atm).sum(axis=1) / (atom.abundance * atm.nh)
        assert np.all(ratio < 1.0 + 1e-2), \
            f"{atom.name}: sum(n_LTE) EXCEEDS the element abundance, max ratio {ratio.max():.4f} (F-001)"
        assert ratio.max() > 0.999, \
            (f"{atom.name}: the model never accounts for the whole element "
             f"(best {ratio.max():.5f}) -- the level set or the stage labelling is wrong")


def test_C3_saha_ratio_is_a_function_of_T_and_ne_only(model):
    """
    (n_i/n_k)* must equal n_e (g_i/2g_k)(h^2/2 pi m_e k T)^{3/2} exp(chi/kT) exactly --
    i.e. the partition functions cancel, so it is immune to F-001 and is legitimately
    computed once outside the Lambda loop (refuted seed lead R-04).
    """
    atoms, atm = model["atoms"], model["atmosphere"]
    for atom in atoms:
        for ic, cont in enumerate(atom.continua):
            i, k = cont.lower_level_index, cont.upper_level_index
            gi, gk = atom.levels[i].g, atom.levels[k].g
            chi = atom.levels[k].energy - atom.levels[i].energy
            for iz in _depths(atm):
                T = atm.temp[iz]
                analytic = (atm.ne_bg[iz] * (gi / (2.0 * gk))
                            * (h_CGS**2 / (2.0 * np.pi * m_e_CGS * kB_CGS * T))**1.5
                            * np.exp(chi / (kB_CGS * T)))
                assert np.isclose(atom.lte_ratios_photoionization[iz, ic], analytic, rtol=1e-9)


# ===========================================================================
# Test D — collision-dominated limit
# ===========================================================================

def _collision_limit_deviation(multiplier):
    """Scale every collisional rate by `multiplier`, solve SE with J = 0, return max |n/n_LTE - 1|."""
    warnings.filterwarnings("ignore")
    with open(CONFIG) as f:
        cfg = json.load(f)
    atm = Atmosphere.from_dict(cfg["atmosphere"])
    atoms = [MultiLevelAtom.from_dict(a) for a in cfg["atoms"]]
    for atom in atoms:
        atom.populations = compute_lte_populations(atom, atm)
        atom.lte_populations = atom.populations.copy()
        atom.compute_doppler_widths(atm, cfg["atmosphere"]["turbulent_velocity"])
        Nz = atm.Ndepth
        atom.Js = np.zeros((Nz, len(atom.lines)))
        atom.Lambda_star_bar = np.zeros((Nz, len(atom.lines)))
        atom.photoionization_rates = np.zeros((Nz, len(atom.continua)))
        atom.recombination_rates = np.zeros((Nz, len(atom.continua)))
        for coll in atom.collisions:
            coll.rates = [r * multiplier for r in coll.rates]
    reference = {a.name: a.lte_populations.copy() for a in atoms}
    solve_SEE(atoms, atm)
    out = {}
    for atom in atoms:
        ref = reference[atom.name]
        significant = (ref / ref.sum(axis=1, keepdims=True)) > 1e-8
        dev = np.abs(atom.populations / np.maximum(ref, 1e-300) - 1.0)
        dev[~significant] = 0.0
        out[atom.name] = dev.max()
    return out


def test_D_collision_dominated_limit_gives_lte():
    """
    Scaling every collisional rate must drive the populations to LTE regardless of the
    radiation field. J is set to ZERO here, so only the collisions can hold the populations
    at Saha-Boltzmann -- a strictly harder test than running with J = B.

    Note the multiplier has to be large: with J = 0 the *unscaled* spontaneous rate A_ji
    still competes with the scaled collisions, so the residual is O(A_ji / (mult * C_ji)).
    The binding case is the transition region (T = 20400 K, n_e = 5.1e10), where
    A(Ly-alpha) = 4.7e8 is large and n_e is small; a multiplier of 1e6 leaves a 134%
    deviation there, which is correct physics, not a bug.
    """
    dev = _collision_limit_deviation(1e10)
    assert dev["H"] < 1e-3, f"H not driven to LTE: max dev {dev['H']:.3e}"
    assert dev["Ca_II"] < 2e-3, f"Ca II not driven to LTE: max dev {dev['Ca_II']:.3e}"


def test_D2_collision_limit_converges_as_one_over_multiplier():
    """
    The approach to LTE must be first order in 1/multiplier, since the residual is set by
    the ratio of the unscaled A_ji to the scaled C_ji. This is a much stronger statement
    than any single tolerance: it confirms the collisional detailed balance is exact and
    that the residual really is the radiative competition and not a systematic error.

    Measured: 1.344 (1e6) -> 1.339e-2 (1e8) -> 1.339e-4 (1e10), i.e. exactly 1/100 per
    two decades, down to the model's own consistency floor.
    """
    d6 = _collision_limit_deviation(1e6)["H"]
    d8 = _collision_limit_deviation(1e8)["H"]
    assert np.isclose(d6 / d8, 100.0, rtol=0.05),         f"collision limit converges as {d6/d8:.1f} per two decades, expected 100"


# ===========================================================================
# Tests E, F — formal solver against closed-form solutions
# ===========================================================================

@pytest.mark.parametrize("mu", [1.0, 0.8, 0.5, 0.2, 0.1])
def test_E_linear_source_is_exact(mu):
    """
    S = a + b*tau  =>  I(0,mu) = a + b*mu exactly (Eddington-Barbier is exact for a linear
    source). The short-characteristic linear solver integrates this without error.
    """
    a, b = 1.0, 2.0
    tau_tot, N = 40.0, 4001
    tau = np.linspace(tau_tot, 0.0, N)
    I = _march(mu, tau, lambda t: a + b * t, a + b * tau_tot + b * mu)
    assert np.isclose(I, a + b * mu, rtol=1e-12)


@pytest.mark.parametrize("dtau", [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3])
def test_E2_no_catastrophic_cancellation(dtau):
    """
    Single step over Delta-tau spanning 1e-10 to 1e3. psi_m and psi_o must match the
    analytic weights, and psi_m + psi_o + exp(-dtau) must equal 1 identically.

    This is where a naive (1 - exp(-x)) implementation loses all precision; psi_lin's
    Taylor branch below dtau = 0.1 handles it (refuted seed lead R-08).
    """
    d = np.array([dtau])
    e = np.exp(-d)
    psi_m, psi_o = psi_lin(e, d)
    assert np.isclose(psi_m[0] + psi_o[0] + e[0], 1.0, rtol=1e-14, atol=1e-14)
    if dtau > 1e-6:                              # analytic form is stable here
        w0 = -np.expm1(-dtau)
        w1 = w0 - dtau * np.exp(-dtau)
        assert np.isclose(psi_m[0], w1 / dtau, rtol=1e-8)
        assert np.isclose(psi_o[0], w0 - w1 / dtau, rtol=1e-8)
    else:                                        # series limit psi -> dtau/2
        assert np.isclose(psi_m[0], dtau / 2.0, rtol=1e-5)
        assert np.isclose(psi_o[0], dtau / 2.0, rtol=1e-5)


def test_E3_psi_o_is_the_local_limit():
    """As Delta-tau -> infinity, I_O -> S_O, i.e. psi_o -> 1 and psi_m -> 0."""
    d = np.array([200.0])
    psi_m, psi_o = psi_lin(np.exp(-d), d)
    assert psi_o[0] > 0.99 and psi_m[0] < 0.01


@pytest.mark.parametrize("tau_tot", [1e-8, 1e-4, 1e-2, 1.0, 10.0])
def test_F_constant_source(tau_tot):
    """I = S(1 - e^-tau) + I0 e^-tau for a constant source function."""
    S0, I0, N = 3.0, 0.5, 101
    tau = np.linspace(tau_tot, 0.0, N)
    I = _march(1.0, tau, lambda t: S0, I0)
    expected = S0 * (1.0 - np.exp(-tau_tot)) + I0 * np.exp(-tau_tot)
    assert np.isclose(I, expected, rtol=1e-10)


# ===========================================================================
# Tests G, H — Lambda* and angular quadrature
# ===========================================================================

def _slab_J_and_lambda(S, mu_nodes, w_nodes, dtau_v):
    """Full sweep over a uniform slab; returns (J, Lambda*_diagonal)."""
    N = len(S)
    J, Lam = np.zeros(N), np.zeros(N)
    for ir, mu in enumerate(mu_nodes):
        if mu > 0:
            order, I = range(0, N), np.array([S[0]])
        else:
            order, I = range(N - 1, -1, -1), np.array([0.0])
        first, S_prev = True, S[0]
        for iz in order:
            if first:
                L, first = np.zeros(1), False
            else:
                I, L = formal_solution(mu, I, dtau_v, np.array([S_prev]),
                                       np.array([S[iz]]), np.ones(1), np.ones(1))
            S_prev = S[iz]
            J[iz] += 0.5 * w_nodes[ir] * I[0]
            Lam[iz] += 0.5 * w_nodes[ir] * L[0]
    return J, Lam


def test_G_lambda_star_is_the_true_diagonal():
    """
    Perturb S at depth k and measure dJ_k/dS_k through the formal solution. It must equal
    Lambda*_kk. An inconsistent Lambda* does not merely slow MALI down -- it converges to
    the wrong populations.
    """
    w, mu_nodes = get_angular_quadrature_1D(6)
    N, tau_tot = 60, 30.0
    dtau_v = tau_tot / (N - 1)
    S = np.linspace(1.0, 2.0, N)
    J0, Lam = _slab_J_and_lambda(S, mu_nodes, w, dtau_v)
    for k in _rng.choice(np.arange(2, N - 2), size=4, replace=False):
        delta = 1e-6 * S[k]
        S2 = S.copy()
        S2[k] += delta
        J1, _ = _slab_J_and_lambda(S2, mu_nodes, w, dtau_v)
        numeric = (J1[k] - J0[k]) / delta
        assert np.isclose(numeric, Lam[k], rtol=1e-2), \
            f"dJ/dS = {numeric:.6f} but Lambda*_kk = {Lam[k]:.6f} at k={k}"


def test_H_isothermal_slab_gives_J_equals_B():
    """
    A static, isothermal, optically thick slab must return J = B at depth and J = B/2 at
    the surface. This is the decisive check on the angular-quadrature normalisation: a
    factor-of-2 error here is invisible in the LTE continuum and lethal in line cores.
    """
    w, mu_nodes = get_angular_quadrature_1D(6)
    N, tau_tot = 200, 60.0
    B = 1.0
    J, _ = _slab_J_and_lambda(np.full(N, B), mu_nodes, w, tau_tot / (N - 1))
    assert np.isclose(J[-1], 0.5 * B, rtol=1e-12), "surface J must be B/2"
    assert np.isclose(J[N // 2], B, rtol=1e-12), "interior J must be B"
    assert np.isclose(J[0], B, rtol=1e-12), "deep J must be B"


def test_H2_gauss_legendre_weights_sum_to_two():
    """Nodes on [-1,1] with sum(w) = 2, no mu = 0 node (so `if ray > 0` is safe)."""
    for n in (2, 4, 5, 6, 8):
        w, mu = get_angular_quadrature_1D(n)
        assert np.isclose(w.sum(), 2.0, rtol=1e-14)
        assert not np.any(mu == 0.0)
        assert np.allclose(np.sort(mu), np.sort(-mu))       # symmetric set


# ===========================================================================
# Test L — conservation and positivity
# ===========================================================================

def test_L_see_conserves_particle_number_and_positivity(model):
    """SEE conserves the explicit active-level pool and returns positive populations."""
    atoms, atm = model["atoms"], model["atmosphere"]
    _build_lte_rates(model)
    solve_SEE(atoms, atm)
    for atom in atoms:
        active_total = atom.lte_populations.sum(axis=1)
        assert np.allclose(atom.populations.sum(axis=1), active_total, rtol=1e-6)
        assert np.all(atom.populations > 0.0), f"{atom.name} has a non-positive population"


def test_L2_lambda_star_bar_is_bounded(model):
    """
    Lambda*-bar = sum 0.5 w psi_o (chi_line/chi_tot) phi must lie in [0,1] by construction,
    since sum(w phi) = 1, psi_o <= 1 and 0 <= chi_line/chi_tot <= 1. Values above 1 mean the
    profile normalisation is broken (F-002) -- which is what the clip at atoms.py:760 was
    hiding. Checked here directly on the integrand.
    """
    atoms, atm, freq = model["atoms"], model["atmosphere"], model["freq"]
    w_ang, mu_nodes = get_angular_quadrature_1D(6)
    nH_ground = atoms[0].populations[:, 0]
    for atom in atoms:
        for line in atom.lines:
            for iz in _depths(atm):
                phi = line_profile(line, atom, iz, freq, atm, nH_ground[iz])
                # Worst case: psi_o = 1 and chi_line/chi_tot = 1 everywhere.
                bound = np.sum(line.freq_weights * phi) * 0.5 * w_ang.sum()
                assert bound <= 1.0 + 1e-9, \
                    f"Lambda*-bar can exceed 1 for {atom.name} {line.lambda0*1e7:.1f} nm"


# ===========================================================================
# Test M — LTE continuum sanity
# ===========================================================================

def test_M_continuum_source_function_equals_planck_where_scattering_is_negligible(model):
    """
    Where thermal absorption dominates over scattering, S_nu = eta/chi must equal B_nu.
    This checks the background opacity bookkeeping and that scattering enters as sigma*J
    rather than sigma*B (refuted seed lead R-13).
    """
    atoms, atm, freq, wq = model["atoms"], model["atmosphere"], model["freq"], model["weights"]
    iz = 0                                   # deepest point: collision/thermal dominated
    atm.J_nu[iz, :] = plank(freq, atm.temp[iz])
    emis, absorp = get_RT_coefficients(iz, freq, wq, atoms, atm)
    B = plank(freq, atm.temp[iz])
    # Restrict to points free of line opacity, in the optical where H- bf/ff dominates.
    wl = (c_CGS / freq) * 1e7
    sel = (wl > 500.0) & (wl < 700.0)
    for atom in atoms:
        for line in atom.lines:
            sel &= ~line.grid_mask
    assert np.any(sel), "no clean continuum points found"
    S = emis[sel] / absorp[sel]
    assert np.allclose(S, B[sel], rtol=1e-6), \
        f"continuum S/B deviates by up to {np.max(np.abs(S/B[sel]-1)):.3e}"


def test_M2_no_negative_opacity_or_emissivity(model):
    """chi and eta must be non-negative everywhere in LTE (refuted seed lead R-15)."""
    atoms, atm, freq, wq = model["atoms"], model["atmosphere"], model["freq"], model["weights"]
    for iz in _depths(atm):
        emis, absorp = get_RT_coefficients(iz, freq, wq, atoms, atm)
        assert np.all(absorp >= 0.0), f"negative opacity at depth {iz}"
        assert np.all(emis >= 0.0), f"negative emissivity at depth {iz}"
        assert np.all(np.isfinite(emis)) and np.all(np.isfinite(absorp))


def test_M3_bound_free_recombination_terms_scale_with_current_ne(monkeypatch):
    """Bound-free emissivity and stimulated extinction are linear in current ne."""
    from types import SimpleNamespace
    import formal_solver as formal_solver_module

    freq = np.array([3.0e14, 6.0e14])
    sigma = np.array([2.0e-18, 5.0e-19])
    n_l, n_u, ratio_bg = 9.0e9, 2.0e8, 4.0
    continuum = SimpleNamespace(lower_level_index=0, upper_level_index=1)
    atom = SimpleNamespace(
        name="X", lines=[], continua=[continuum],
        populations=np.array([[n_l, n_u]]),
        lte_ratios_photoionization=np.array([[ratio_bg]]),
        photoionization_alphas=np.array([sigma]),
    )
    atmosphere = SimpleNamespace(
        Ndepth=1, temp=np.array([8000.0]), nh=np.array([1.0e15]),
        ne_bg=np.array([1.0e12]), ne=np.array([1.0e12]),
    )

    zeros = lambda iz, grid, atoms, atmos: (np.zeros_like(grid), np.zeros_like(grid))
    monkeypatch.setattr(formal_solver_module, "add_background_opacity", zeros)

    eta_1, chi_1 = formal_solver_module.get_RT_coefficients(
        0, freq, np.ones_like(freq), [atom], atmosphere)
    stimulated_1 = sigma*n_l - chi_1

    atmosphere.ne[:] = 2.0*atmosphere.ne_bg
    eta_2, chi_2 = formal_solver_module.get_RT_coefficients(
        0, freq, np.ones_like(freq), [atom], atmosphere)
    stimulated_2 = sigma*n_l - chi_2

    assert np.allclose(eta_2, 2.0*eta_1, rtol=1e-14)
    assert np.allclose(stimulated_2, 2.0*stimulated_1, rtol=1e-12)


def test_M4_three_body_recombination_scales_with_ne_squared():
    """A collision-only ionization pair obeys current-ne detailed balance."""
    from types import SimpleNamespace

    level_0 = SimpleNamespace(energy=0.0, g=1.0, ionization=0)
    level_1 = SimpleNamespace(energy=0.0, g=1.0, ionization=1)
    collision = SimpleNamespace(
        lower_level_index=0, upper_level_index=1, type="CI",
        temperatures=np.array([5000.0, 10000.0]),
        rates=np.array([1.0e-8, 1.0e-8]),
    )
    atom = SimpleNamespace(
        name="X", levels=[level_0, level_1], collisions=[collision],
        lines=[], continua=[], lte_populations=np.array([[4.0, 1.0]]),
        populations=np.array([[4.0, 1.0]]),
        Js=np.zeros((1, 0)), Lambda_star_bar=np.zeros((1, 0)),
        photoionization_rates=np.zeros((1, 0)),
        recombination_rates=np.zeros((1, 0)),
    )

    ne_bg = 1.0e12
    common = dict(atom=atom, k=0, Tk=8000.0, kT=kB_CGS*8000.0,
                  N_active_k=5.0, n_h0=1.0e15, n_proton=1.0e12,
                  old_pops_k=np.array([4.0, 1.0]),
                  atmosphere_ne_bg_k=ne_bg)
    pops_bg = solve_atom(ne=ne_bg, **common)
    pops_2bg = solve_atom(ne=2.0*ne_bg, **common)

    assert np.isclose(pops_bg[0] / pops_bg[1], 4.0, rtol=1e-12)
    assert np.isclose(pops_2bg[0] / pops_2bg[1], 8.0, rtol=1e-12)


def test_M5_hydrogen_collision_coefficients_use_lw_units_and_colliders():
    """CH/CP are Lightweaver m^3/s tables multiplied by the matching cm^-3 collider."""
    from types import SimpleNamespace

    levels = [SimpleNamespace(energy=0.0, g=1.0, ionization=0),
              SimpleNamespace(energy=0.0, g=1.0, ionization=0)]
    coefficient = 2.5e-15
    common = dict(
        name="X", levels=levels, lines=[], continua=[],
        lte_populations=np.array([[1.0, 1.0]]),
        populations=np.array([[1.0, 1.0]]),
        Js=np.zeros((1, 0)), Lambda_star_bar=np.zeros((1, 0)),
        photoionization_rates=np.zeros((1, 0)),
        recombination_rates=np.zeros((1, 0)),
    )

    for collision_type, n_h0, n_proton, matrix_index, collider in (
            ("CH", 3.0e10, 7.0e8, (0, 1), 3.0e10),
            ("CP", 3.0e10, 7.0e8, (1, 0), 7.0e8)):
        collision = SimpleNamespace(
            lower_level_index=0, upper_level_index=1, type=collision_type,
            temperatures=np.array([5000.0, 10000.0]),
            rates=np.array([coefficient, coefficient]))
        atom = SimpleNamespace(collisions=[collision], **common)
        solve_atom(
            atom, k=0, Tk=8000.0, ne=1.0e12, kT=kB_CGS*8000.0,
            N_active_k=2.0, n_h0=n_h0, n_proton=n_proton,
            old_pops_k=np.array([1.0, 1.0]), atmosphere_ne_bg_k=1.0e12)
        assert np.isclose(atom.C_matrix_all[0][matrix_index],
                          coefficient * collider * 1e6, rtol=1e-14)


def test_M6_hydrogen_colliders_follow_explicit_model_or_mode_consistent_lte():
    """Use explicit H populations; otherwise update LTE H only in full-nlte mode."""
    from types import SimpleNamespace

    levels = [SimpleNamespace(energy=2.0, ionization=0),
              SimpleNamespace(energy=0.0, ionization=0),
              SimpleNamespace(energy=3.0, ionization=1)]
    hydrogen = SimpleNamespace(
        Z=1, levels=levels, populations=np.array([[2.0, 7.0, 3.0]]))
    atmosphere = SimpleNamespace(
        ne_bg=np.array([100.0]), nh=np.array([10.0]),
        bg_species={'n_HI_per_U': np.array([4.0]), 'n_HII': np.array([2.0])})

    assert _hydrogen_collider_densities(
        [hydrogen], atmosphere, 0, "nlte", 200.0) == (7.0, 3.0)
    assert np.allclose(_hydrogen_collider_densities(
        [], atmosphere, 0, "eos", 200.0), (8.0, 2.0))
    assert np.allclose(_hydrogen_collider_densities(
        [], atmosphere, 0, "delta", 200.0), (8.0, 2.0))
    assert np.allclose(_hydrogen_collider_densities(
        [], atmosphere, 0, "nlte", 200.0), (80.0/9.0, 10.0/9.0))


def test_M7_seaton_retains_half_integer_power():
    """The Seaton POWER=1.5 exponent must remain 1.5 rather than truncate to 1."""
    value = _seaton(1.0, 1.0, 1.5, 1.0, np.array([2.0]))[0]
    assert np.isclose(value, 0.5**1.5, rtol=1e-15)


def test_M8_failed_linear_solve_cannot_report_convergence(model, monkeypatch):
    """A failed SE solve restores its input state and returns an explicit failure."""
    import atoms as atoms_module

    atoms, atmosphere = model["atoms"], model["atmosphere"]
    populations_before = {atom.name: atom.populations.copy() for atom in atoms}
    ne_before = atmosphere.ne.copy()

    def fail_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("injected singular rate matrix")

    monkeypatch.setattr(atoms_module, "solve_atom", fail_solve)
    change, succeeded = solve_SEE(
        atoms, atmosphere, electron_mode="eos", return_status=True)

    assert not succeeded
    assert np.isinf(change)
    assert np.array_equal(atmosphere.ne, ne_before)
    for atom in atoms:
        assert np.array_equal(atom.populations, populations_before[atom.name])


# ===========================================================================
# Test J — Lambda*-off equivalence (the decisive MALI test)
# ===========================================================================

def test_J_preconditioning_reduces_to_lambda_iteration(model):
    """
    The MALI preconditioning must be an exact reformulation, not an approximation: at a
    fixed point, the net rate n_i R_ij - n_j R_ji must be independent of Lambda*.

    Algebraically, with S_l = n_j A_ji / (n_i B_ij - n_j B_ji),
        n_i B_ij (Jbar - L S_l) - n_j [A_ji(1-L) + B_ji(Jbar - L S_l)]
      = (n_i B_ij - n_j B_ji) Jbar - n_j A_ji
    for ANY L -- the Lambda* terms cancel identically. So solving the SE with Lambda* = 0
    and then re-solving with a large Lambda*, feeding back the first solution as the
    populations that define S_old, must return the same populations.

    If they differ, the preconditioning is wrong and MALI converges to the wrong answer.
    This is done algebraically rather than by running two full Lambda iterations, so it
    isolates the preconditioning from convergence-rate effects.
    """
    atoms, atm = model["atoms"], model["atmosphere"]
    _build_lte_rates(model)

    # (1) plain Lambda-iteration: Lambda* == 0
    for atom in atoms:
        atom.Lambda_star_bar = np.zeros_like(atom.Lambda_star_bar)
        atom.populations = atom.lte_populations.copy()
    solve_SEE(atoms, atm)
    plain = {a.name: a.populations.copy() for a in atoms}
    ne_plain = atm.ne.copy()

    # (2) MALI: strong Lambda*, started from the fixed point found above
    for atom in atoms:
        atom.Lambda_star_bar = np.full_like(atom.Lambda_star_bar, 0.9)
        atom.populations = plain[atom.name].copy()
    atm.ne = ne_plain.copy()
    solve_SEE(atoms, atm)

    for atom in atoms:
        dev = np.max(np.abs(atom.populations / np.maximum(plain[atom.name], 1e-300) - 1.0))
        # solve_SEE runs its local (populations <-> ne) loop to tolerance=1e-5 with 0.5
        # damping, so the two solutions can only agree to about that level. Measured
        # residual is 1.7e-6, i.e. below the solver's own convergence tolerance -- the
        # preconditioning itself contributes nothing detectable. test_J2 checks the
        # underlying algebraic identity to 1e-10, free of any solver tolerance.
        assert dev < 1e-4, \
            (f"{atom.name}: MALI with Lambda*=0.9 moved the fixed point by {dev:.3e}; "
             "the preconditioning is not an exact reformulation")


def test_J2_lambda_star_cancels_in_the_net_rate(model):
    """
    Direct algebraic check of the identity above, on the parsed atomic data, over a sweep
    of Lambda* values including the clip boundary at atoms.py:760.
    """
    atom = model["atoms"][1]                     # Ca II
    line = atom.lines[0]
    n_l, n_u = 1.0e6, 3.0e2
    S_l = (n_u * line.Aul) / (n_l * line.Blu - n_u * line.Bul)
    # J_bar must exceed Lambda* S_l or the max(J_eff, 0) clamp at atoms.py:775 fires and
    # the identity no longer applies -- which is itself worth knowing: that clamp is only
    # inert while the line is not too far from radiative equilibrium.
    J_bar = 1.5 * S_l
    reference = (n_l * line.Blu - n_u * line.Bul) * J_bar - n_u * line.Aul
    for L in (0.0, 0.1, 0.5, 0.9, 0.9999999):
        J_eff = J_bar - L * S_l
        assert J_eff > 0.0, "clamp would fire; test setup invalid"
        net = n_l * (line.Blu * J_eff) - n_u * (line.Aul * (1.0 - L) + line.Bul * J_eff)
        assert np.isclose(net, reference, rtol=1e-10), f"net rate depends on Lambda* at L={L}"


# ===========================================================================
# Test K — grid independence
# ===========================================================================

def _cached_opacity(model):
    """Opacities depend only on depth, not on the ray. Cache them once."""
    atoms, atm, freq, wq = model["atoms"], model["atmosphere"], model["freq"], model["weights"]
    chi = np.zeros((atm.Ndepth, len(freq)))
    eta = np.zeros((atm.Ndepth, len(freq)))
    for iz in range(atm.Ndepth):
        atm.J_nu[iz, :] = plank(freq, atm.temp[iz])
    for iz in range(atm.Ndepth):
        e, a = get_RT_coefficients(iz, freq, wq, atoms, atm)
        eta[iz, :], chi[iz, :] = e, a
    return eta, chi


def _sweep_J(eta, chi, zgrid, n_gauss):
    """Mean intensity on the real stratification for a given ray count."""
    w, mus = get_angular_quadrature_1D(n_gauss)
    Nz, Nnu = chi.shape
    J = np.zeros((Nz, Nnu))
    S = eta / np.maximum(chi, 1e-100)
    for ir, mu in enumerate(mus):
        if mu > 0:
            order, I = range(0, Nz), S[0, :].copy()
        else:
            order, I = range(Nz - 1, -1, -1), np.zeros(Nnu)
        prev = None
        for iz in order:
            if prev is not None:
                dz = abs(zgrid[iz] - zgrid[prev])
                I, _ = formal_solution(mu, I, dz, eta[prev, :], eta[iz, :],
                                       chi[prev, :], chi[iz, :])
            prev = iz
            J[iz, :] += 0.5 * w[ir] * I
    return J


def _jbar_for_rays(model, n_gauss, eta, chi):
    """Profile-weighted J-bar per (depth, line) for a given ray count."""
    atm, freq = model["atmosphere"], model["freq"]
    J = _sweep_J(eta, chi, atm.zgrid, n_gauss)
    nH_ground = model["atoms"][0].populations[:, 0]
    out = {}
    for atom in model["atoms"]:
        arr = np.zeros((atm.Ndepth, len(atom.lines)))
        for iz in range(atm.Ndepth):
            for il, line in enumerate(atom.lines):
                phi = line_profile(line, atom, iz, freq, atm, nH_ground[iz])
                arr[iz, il] = np.sum(line.freq_weights * J[iz, :] * phi)
        out[atom.name] = arr
    return out


def test_K_angular_quadrature_converges(model):
    """
    Successive refinement of the ray set must make J-bar -- the profile-weighted mean
    intensity that actually feeds the radiative rates -- converge.

    J itself is NOT a meaningful target: in the extreme UV it falls to ~1e-39, where
    relative changes are noise. Nor is the single worst H point (the 95 nm Lyman line at
    the 1e5 K top, J-bar ~ 1e-8), which does not converge with ray count at all -- that is
    the under-resolved line window of audit F-003, not a quadrature problem. Ca II, the
    benchmark target, is the meaningful check.

    What this test asserts is convergence, not a tolerance at the production setting,
    because the production setting is genuinely too coarse: even with the hemispheric
    quadrature of F-014, Ca II J-bar still moves 1.7% between 6 and 10 rays. See F-014 --
    n_gaus should be raised to >= 12 for quantitative work.
    """
    eta, chi = _cached_opacity(model)
    b6 = _jbar_for_rays(model, 6, eta, chi)
    b12 = _jbar_for_rays(model, 12, eta, chi)
    b24 = _jbar_for_rays(model, 24, eta, chi)

    def change(a, b):
        r = np.abs(b["Ca_II"] / np.maximum(a["Ca_II"], 1e-300) - 1.0)
        return np.median(r), r.max()

    med_coarse, max_coarse = change(b6, b12)
    med_fine, max_fine = change(b12, b24)
    assert med_fine < med_coarse, \
        f"Ca II J-bar not converging: median change {med_coarse:.3e} -> {med_fine:.3e}"
    assert max_fine < max_coarse, \
        f"Ca II J-bar not converging: max change {max_coarse:.3e} -> {max_fine:.3e}"
    assert med_fine < 1e-3, \
        f"Ca II J-bar median still {med_fine:.3e} at 24 rays"


def test_K3_hemispheric_quadrature_beats_full_range(model):
    """
    Regression guard for F-014: Gauss-Legendre applied per hemisphere must be more accurate
    than a single rule spanning [-1,1], because I(mu) is discontinuous at mu = 0 (no
    incoming radiation at the top boundary, finite outgoing intensity).

    Compared at equal ray count against a common high-resolution reference.
    """
    eta, chi = _cached_opacity(model)

    def full_range(n):
        if n % 2 == 1:
            n += 1
        mu, w = np.polynomial.legendre.leggauss(n)
        return w, mu

    atm, freq = model["atmosphere"], model["freq"]
    nH_ground = model["atoms"][0].populations[:, 0]

    def jbar(w, mus):
        Nz, Nnu = chi.shape
        J = np.zeros((Nz, Nnu))
        S = eta / np.maximum(chi, 1e-100)
        for ir, mu in enumerate(mus):
            if mu > 0:
                order, I = range(0, Nz), S[0, :].copy()
            else:
                order, I = range(Nz - 1, -1, -1), np.zeros(Nnu)
            prev = None
            for iz in order:
                if prev is not None:
                    I, _ = formal_solution(mu, I, abs(atm.zgrid[iz] - atm.zgrid[prev]),
                                           eta[prev, :], eta[iz, :], chi[prev, :], chi[iz, :])
                prev = iz
                J[iz, :] += 0.5 * w[ir] * I
        atom = model["atoms"][1]                      # Ca II
        arr = np.zeros((atm.Ndepth, len(atom.lines)))
        for iz in range(atm.Ndepth):
            for il, line in enumerate(atom.lines):
                phi = line_profile(line, atom, iz, freq, atm, nH_ground[iz])
                arr[iz, il] = np.sum(line.freq_weights * J[iz, :] * phi)
        return arr

    reference = jbar(*get_angular_quadrature_1D(48))
    hemi = jbar(*get_angular_quadrature_1D(8))
    full = jbar(*full_range(8))
    err_hemi = np.median(np.abs(hemi / reference - 1.0))
    err_full = np.median(np.abs(full / reference - 1.0))
    assert err_hemi < err_full, \
        (f"hemispheric quadrature ({err_hemi:.3e}) should beat full-range "
         f"({err_full:.3e}) at equal ray count")


def test_K2_frequency_grid_refinement_converges_second_order():
    """
    Refining the per-line frequency grid must drive int(phi dnu) -> 1 at second order,
    confirming the residual after F-002 is ordinary trapezoid error and nothing structural.
    """
    warnings.filterwarnings("ignore")
    from scipy.special import wofz
    with open(CONFIG) as f:
        base = json.load(f)
    atm = Atmosphere.from_dict(base["atmosphere"])
    errors = []
    for scale in (1, 2, 4):
        cfg = json.loads(json.dumps(base))
        for a in cfg["atoms"]:
            for ln in a["lines"]:
                ln["quadrature"]["N_lambda"] *= scale
        atoms = [MultiLevelAtom.from_dict(a) for a in cfg["atoms"]]
        for atom in atoms:
            atom.populations = compute_lte_populations(atom, atm)
            atom.lte_populations = atom.populations.copy()
            atom.compute_doppler_widths(atm, cfg["atmosphere"]["turbulent_velocity"])
        freq, _ = create_frequency_grid(atoms, atmosphere=Atmosphere.from_dict(cfg["atmosphere"]), v_turb=cfg["atmosphere"]["turbulent_velocity"])
        compute_line_frequency_weights(atoms, freq)
        worst = 0.0
        for atom in atoms:
            for il, line in enumerate(atom.lines):
                gamma = sum(n.get("gamma", 0.0) for n in line.broadening.natural)
                dnuD = atom.doppler_widths[0, il]
                a = gamma / (4.0 * np.pi * dnuD)
                phi = wofz((freq - line.nu0) / dnuD + 1j * a).real / (np.sqrt(np.pi) * dnuD)
                worst = max(worst, abs(np.sum(phi[line.grid_mask]
                                              * line.freq_weights[line.grid_mask]) - 1.0))
        errors.append(worst)
    assert errors[0] > errors[1] > errors[2], f"grid refinement does not converge: {errors}"
    assert errors[1] / errors[2] > 2.5, \
        f"convergence order too low: errors {errors} (expected ~4x per doubling)"


# ===========================================================================
# Cross-validation against Lightweaver (the reference implementation)
# ===========================================================================

# Scoped to the one test that needs it. At module level, `importorskip` skips the WHOLE
# module, so with Lightweaver absent the 45 analytic tests below silently did not run --
# the suite reported "1 skipped" and looked green.
def test_broadening_matches_lightweaver(model):
    """
    Compare the total damping rate Gamma for every line against Lightweaver's own
    Broadening.py, evaluated on the same stratification with the same n_H(ground).

    This is a numerical cross-check of Unsold, quadratic Stark, linear Stark and the
    natural width together, against the reference implementation -- not just a
    dimensional argument.

    Requires Lightweaver; skipped when it is not installed.

    Ca II tolerance is 1%: the config's Grad (1.48-1.50e8) differs from the value implied
    by its own f-values and from Lightweaver's atom file (1.55-1.58e8) by ~4.5%, which is
    audit finding F-010. That shifts the total by ~0.4%.
    """
    lightweaver = pytest.importorskip("lightweaver", reason="Lightweaver not installed")
    from lightweaver.rh_atoms import H_6_atom, CaII_atom

    atoms, atm = model["atoms"], model["atmosphere"]
    lwatm = lightweaver.Atmosphere.make_1d(
        scale=lightweaver.ScaleType.Geometric,
        depthScale=(atm.zgrid * 1e-2)[::-1].copy(),
        temperature=atm.temp[::-1].copy(),
        vlos=np.zeros(atm.Ndepth),
        vturb=np.full(atm.Ndepth, model["v_turb"] * 1e-2),
        ne=(atm.ne * 1e6)[::-1].copy(),
        nHTot=(atm.nh * 1e6)[::-1].copy())
    lwatm.quadrature(5)
    rs = lightweaver.RadiativeSet([H_6_atom(), CaII_atom()])
    rs.set_active('H', 'Ca')
    eqPops = rs.compute_eq_pops(lwatm)
    nH_ground_lw = eqPops['H'][0, -1] / 1e6           # deepest point, m^-3 -> cm^-3

    tolerance = {'H': 2e-3, 'Ca_II': 1e-2}
    checked = 0
    for lwatom, ourname in ((rs.activeAtoms[0], 'H'), (rs.activeAtoms[1], 'Ca_II')):
        ouratom = next(a for a in atoms if a.name == ourname)
        for ln in lwatom.lines:
            dist, ourline = min(((abs(l.lambda0 * 1e7 - ln.lambda0), l) for l in ouratom.lines),
                                key=lambda t: t[0])
            if dist > 0.5:
                continue
            gamma_lw = 0.0
            for b in list(ln.broadening.natural) + list(ln.broadening.elastic):
                gamma_lw += np.atleast_1d(b.broaden(lwatm, eqPops))[-1]
            gamma_ours = line_damping(ourline, ouratom, 0, atm, nH_ground_lw)
            assert np.isclose(gamma_ours, gamma_lw, rtol=tolerance[ourname]), \
                (f"{ourname} {ln.lambda0:.2f} nm: Gamma = {gamma_ours:.4e} (firtez) vs "
                 f"{gamma_lw:.4e} (Lightweaver), ratio {gamma_ours/gamma_lw:.4f}")
            checked += 1
    assert checked >= 14, f"only {checked} lines cross-checked against Lightweaver"
