# Segmented Spacetime as a Natural Regulator of Superradiant Instabilities

Authors: Carmen Wrede & Lino Casu

Affiliation: Independent Research Group "Segmented Spacetime"

**Date:** 2025

#### **Abstract**

We present a computational and analytical study demonstrating that a Segmented Spacetime (SSZ) framework naturally suppresses superradiant amplification in rotating systems analogous to the "blackhole bomb" mechanism predicted by Press and Teukolsky (1972). Using a  $\varphi$ -based discretization of spacetime curvature, we show that local segmentation acts as a distributed damping term that limits exponential amplitude growth. Simulations of the SSZ analog system reveal a substantial reduction of unstable modes and gain factors by up to ten orders of magnitude compared to baseline relativistic configurations. The results suggest that spacetime segmentation itself may constitute a stabilizing boundary condition preventing uncontrolled energy extraction from rotating gravitational systems.

## 1. Introduction

Rotating black holes in general relativity permit *superradiant scattering*, where incident waves with frequencies  $\omega < m\Omega$  extract rotational energy from the spacetime metric.

Press and Teukolsky (1972) proposed that enclosing such a system within a reflecting cavity would result in an exponentially growing instability — the so-called *black-hole bomb*.

However, real astrophysical black holes show no evidence of runaway amplification. The *Segmented Spacetime Model (SSZ)* offers an alternative explanation:

space itself is not continuous, but divided into discrete  $\varphi$  -segments whose local density increases with curvature. Each segment boundary introduces a small amplitude reduction and phase twist. This microscopic segmentation behaves as a natural dissipative lattice embedded in spacetime.

# 2. Methodology

We constructed a numerical analog of the Press-Teukolsky setup, extended with SSZ segmentation:

$$G_{ ext{SSZ}} = \expigg[\int \gamma_{ ext{loc}}\,dsigg] \cdot \prod_{k=1}^K e^{-\lambda_A \sigma( heta_k)} \cdot \mathcal{R}(1-\mathcal{K})$$

where: -  $\lambda_A$  controls amplitude damping per segment,

- $\lambda_{\omega}$  encodes phase torsion,
- K is the number of spatial segments, and
- $\Omega_0$  denotes the rotation parameter of the analog cavity.

Parameter scans were performed across grids:

$$\lambda_A, \lambda_{\varphi} = [0.00...0.05], \quad K = [8, 16, 32, 64], \quad \Omega_0 = [0.2, 0.3, 0.4].$$

Each configuration computed the gain G , instability counts, and the stabilization index:

$$S = rac{\Delta ext{unstable}}{ ext{base unstable}} + \langle \Delta \log G 
angle.$$

All simulations used the *Perfect Pair* computation scheme, with numerical invariants verified to machine precision (error  $\leq 10^{-6}$ ).

#### 3. Results

#### 3.1 Baseline behavior

Without segmentation ( $\lambda_A=0$  ), amplitude grows exponentially by  $10^{11}$  within two roundtrips — consistent with a self-reinforcing superradiant mode.

#### 3.2 SSZ stabilization

Introducing segmentation damping ( $\lambda_A \geq 0.02$  ) produces dramatic effects: - Unstable mode count decreases by up to **15** ( $\Delta {
m unstable}=-15$  ).

- Average gain drops by **pprox 10 orders of magnitude** ( $\Delta \log G = -9.6$  ).
- Stabilization index S pprox -10.5 for the optimal configuration  $[\lambda_A=0.05, K=64, \Omega_0=0.2]$  .

## 3.3 Correlation with GR parameters

Cross-analysis with GR reference modes  $(\omega, m) = (0.1 - 0.3, 2 - 4)$  shows:

$$r(G,S) \approx 0.9$$
,  $r(\text{segment density}, S) \approx -0.36$ .

Hence, stronger GR-like gain corresponds to stronger SSZ suppression.

#### 3.4 Parameter-space maps

Heatmaps for K=64 show that stabilization strength grows nearly linearly with  $\lambda_A$  , while  $\lambda_\varphi$  remains secondary.

For  $\lambda_A \geq 0.05$  , S reaches pprox –10, independent of  $\Omega_0$  .

# 4. Discussion

The SSZ model reproduces classical superradiance in the continuous limit and transitions smoothly into a stabilized regime once segmentation is introduced.

The mechanism acts as a *distributed boundary layer*: each segment reabsorbs a fraction of the wave energy, disrupting phase coherence and exponential feedback.

This provides a physical explanation for the absence of observed runaway amplification in real astrophysical systems: local segment density near the ergosphere may exceed the threshold where G<1, halting the feedback loop.

Thus, SSZ introduces a geometric damping mechanism that replaces artificial mirrors with the intrinsic structure of space itself.

Moreover, this suppression aligns with GR's global invariants: energy is conserved, but redistributed into curvature-bound micro-modes instead of escaping macroscopically.

# **5. Conclusion**

Segmented spacetime geometries inherently stabilize rotating systems against superradiant runaway. The presence of φ-segments acts as a natural regulator, reducing amplification by many orders of magnitude without invoking external damping or quantum effects.

This result supports the hypothesis that spacetime's granular architecture is self-regulating and may resolve the absence of astrophysical "black-hole bombs."

# **References**

- Press, W. H. & Teukolsky, S. A. (1972). *Floating orbits, superradiant scattering and the black-hole bomb.* Nature, 238, 211–212. 1.
- Zel'dovich, Y. B. (1971). *Generation of waves by a rotating body.* JETP Letters, 14, 180. 2.
- Casu, L. & Wrede, C. (2025). *Segmented Spacetime and the Natural Boundary of Black Holes: Implications for the Cosmic Censorship Conjecture.* ResearchHub Preprint. 3.
- Casu, L. & Wrede, C. (2025). *Segmented Spacetime Solution to the Paradox of Singularities.* Independent Publication. 4.
- Teukolsky, S. A. (1973). *Perturbations of a rotating black hole.* ApJ, 185, 635. 5.
- Casu, L. & Wrede, C. (2025). *Black-Hole Bomb Simulations under SSZ Conditions.* Internal Dataset (Perfect Pair Script Results). 6.