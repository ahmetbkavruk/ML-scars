# ML-scars
We investigate whether classical machine learning can guide a search for Hamiltonians with quantum many-body scars. The initial benchmark is a deformation of the Rydberg-blockaded PXP model. No scar eigenvectors or scar algebra are supplied to the search; the core project targets revivals from a specified initial state.

The project builds on the following Hamiltonian:

$$
\begin{aligned}
H(\boldsymbol{\lambda}) ={}&
J\sum_i K_i \Big[
1+\lambda_2(\sigma^z_{i-2}+\sigma^z_{i+2})\\
&+\lambda_3(\sigma^z_{i-3}+\sigma^z_{i+3})\\
&+\lambda_{22}\sigma^z_{i-2}\sigma^z_{i+2}
\Big]\\
&+J\Delta\sum_i n_i
+JV\sum_i n_i n_{i+2}.
\end{aligned}
$$

Here,

$$
K_i=P_{i-1}\sigma_i^x P_{i+1},
\qquad
P_i=\frac{1+\sigma_i^z}{2},
\qquad
n_i=\frac{1-\sigma_i^z}{2},
$$

and $\boldsymbol{\lambda}=(\lambda_2,\lambda_3,\lambda_{22},\Delta,V)$.

We impose periodic boundary conditions, with all site indices understood modulo $L$.

## References

1. S. Choi et al., *Emergent SU(2) Dynamics and Perfect Quantum Many-Body Scars*, [Phys. Rev. Lett. **122**, 220603 (2019)](https://doi.org/10.1103/PhysRevLett.122.220603).

2. N. Swain et al., *Engineering many-body quantum Hamiltonians with nonergodic properties using quantum Monte Carlo*, [Phys. Rev. B **111**, 224201 (2025)](https://doi.org/10.1103/PhysRevB.111.224201).

3. S. Moudgalya and O. I. Motrunich, *Exhaustive Characterization of Quantum Many-Body Scars Using Commutant Algebras*, [Phys. Rev. X **14**, 041069 (2024)](https://doi.org/10.1103/PhysRevX.14.041069).

4. J. Ren et al., *ScarFinder: A Detector of Optimal Scar Trajectories in Quantum Many-Body Dynamics*, [PRX Quantum **6**, 040332 (2025)](https://doi.org/10.1103/8g8w-nkwx).
