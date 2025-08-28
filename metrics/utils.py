"""
Utility functions for QDN metrics calculations.

This module provides implementations of key QDN metrics:
- Fidelity (based on Qiskit's state_fidelity)
- Stream Braid Quotient (SBQ)
- Resonance Integrity Quotient (RIQ)

These metrics quantify QDN performance, resilience, and adaptability 
under various noise conditions.
"""

import numpy as np
from qiskit.quantum_info import state_fidelity

def calculate_fidelity(simulated_state, reference_state):
    """
    Calculate fidelity between simulated and reference quantum states.
    
    Args:
        simulated_state: Statevector or density matrix from simulation
        reference_state: Ideal reference state (e.g., from GHZState)
        
    Returns:
        float: Fidelity value between 0 and 1
    """
    return state_fidelity(simulated_state, reference_state)

def sbq(path, graph):
    """
    Stream Braid Quotient: Measures entanglement path density in topological encodings.
    
    Higher values indicate better resilience to local noise via distributed routing.
    
    Args:
        path: List of nodes representing the selected computational path
        graph: NetworkX graph representing the braid/entanglement structure
        
    Returns:
        float: SBQ value between 0 and 1
    """
    if graph.number_of_edges() == 0:
        return 0
    
    # Length of path divided by total edges in graph
    return len(path) / graph.number_of_edges()

def riq(noise_profiles):
    """
    Resonance Integrity Quotient: Measures stability across sampled noise profiles.
    
    Low variance indicates effective damping parameter refinement.
    
    Args:
        noise_profiles: Array of noise measurements (e.g., damped amplitudes over time)
        
    Returns:
        float: RIQ value between 0 and 1 (higher is better)
    """
    if len(noise_profiles) <= 1:
        return 1.0
    
    # 1 minus standard deviation of noise profiles
    return 1.0 - np.std(noise_profiles)

def combined_metric(sbq_value, riq_value, weights=(0.5, 0.5)):
    """
    Calculate a combined metric from SBQ and RIQ with optional weighting.
    
    Args:
        sbq_value: Stream Braid Quotient value
        riq_value: Resonance Integrity Quotient value
        weights: Tuple of weights for (SBQ, RIQ), should sum to 1
        
    Returns:
        float: Combined metric value between 0 and 1
    """
    return weights[0] * sbq_value + weights[1] * riq_value

def check_thresholds(fidelity=None, sbq_value=None, riq_value=None):
    """
    Check if metrics meet the recommended thresholds.
    
    Args:
        fidelity: Fidelity value (optional)
        sbq_value: SBQ value (optional)
        riq_value: RIQ value (optional)
        
    Returns:
        dict: Dictionary of metrics with boolean indicating if threshold is met
    """
    results = {}
    
    if fidelity is not None:
        results['fidelity'] = {'value': fidelity, 'meets_threshold': fidelity > 0.95}
    
    if sbq_value is not None:
        results['sbq'] = {'value': sbq_value, 'meets_threshold': sbq_value > 0.5}
    
    if riq_value is not None:
        results['riq'] = {'value': riq_value, 'meets_threshold': riq_value > 0.8}
        
    if sbq_value is not None and riq_value is not None:
        combined = sbq_value * riq_value
        results['combined'] = {'value': combined, 'meets_threshold': combined > 0.9}
    
    return results
