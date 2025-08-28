# QDN Metrics Overview

This directory outlines key metrics used in Quantum Drift Nexus (QDN) simulations. These quantify system performance, resilience, and adaptability under noise, aligning with the blueprint's focus on treating decoherence as a computational resource. Equations are provided for reproducibility, with references to Qiskit functions where applicable.

## 1. Fidelity (State Fidelity)
- **Description**: Measures how closely a simulated quantum state matches an ideal (noiseless) reference state. In QDN, this validates holographic error correction and adaptive protocols, with boosts indicating effective drift navigation.
- **Use Case**: Baseline vs. adaptive comparisons in circuits like Bell states, QFT, or GHZ (e.g., +4-11% gains in Phase 1).
- **Equation**:
  $$
  F(\rho, \sigma) = \left( \text{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2
  $$
  Where $\rho$ is the simulated density matrix, and $\sigma$ is the ideal.
- **Implementation**: Use Qiskit's `state_fidelity` from `qiskit.quantum_info`.
- **Thresholds**: Target >0.95 for scalable operations; dips trigger bio-adaptive mutations.

## 2. Stream Braid Quotient (SBQ)
- **Description**: A proxy for entanglement path density in topological encodings, assessing how "braided" (interconnected) computational paths are. Higher SBQ implies better resilience to local noise via distributed routing in the DVSP.
- **Use Case**: In amplitude damping models, SBQ guides dynamic path selection, leveraging drifts for stable trajectories.
- **Equation** (Graph-Based Proxy):
  $$
  SBQ = \frac{L_p}{E_G}
  $$
  Where $L_p$ is the length of the selected path (e.g., shortest stable route), and $E_G$ is the total edges in the braid graph (e.g., via NetworkX).
- **Implementation**: Custom function in simulations; normalize to [0,1] for quotients.
- **Thresholds**: >0.5 for robust braiding; used in optimization loops to mutate paths.

## 3. Resonance Integrity Quotient (RIQ)
- **Description**: Measures resonance stability across sampled noise profiles, quantifying how well the unified wave equation maintains coherence amid drifts. Low variance indicates effective damping parameter (α) refinement.
- **Use Case**: Preempts fidelity drops by sampling environmental profiles; integrates with Drift Archive for predictive adjustments.
- **Equation**:
  $$
  RIQ = 1 - \sigma(N_p)
  $$
  Where $\sigma(N_p)$ is the standard deviation of noise profiles $N_p$ (e.g., damped amplitudes over steps).
- **Implementation**: NumPy-based; combine with SBQ for composite scores (e.g., RIQ * SBQ > 0.9 triggers selection).
- **Thresholds**: >0.8 for integrity; below prompts DVSP rerouting.

These metrics are extensible—feel free to PR refinements! For examples, see `qdn_scaling_demo.ipynb` (Fidelity focus) and `qdn_amplitude_damping.ipynb` (SBQ/RIQ integration).

References: Qiskit Documentation, arXiv:quant-ph papers on topological QC.
