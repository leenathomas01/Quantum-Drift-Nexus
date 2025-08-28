"""
Quantum Drift Nexus - Amplitude Damping Simulation

This module demonstrates the QDN approach to handling amplitude damping noise,
implementing the dynamic path selection via Stream Braid Quotient (SBQ) and
Resonance Integrity Quotient (RIQ) metrics.

This is part of Phase II of the QDN project, focusing on more complex noise models
and their integration with the holographic error correction approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

# Import QDN metrics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.utils import sbq, riq, combined_metric, check_thresholds


class DVSPRouter:
    """
    Dynamic Variable Stream Pathfinder (DVSP) Router
    
    Implements the adaptive pathfinding protocol from the QDN blueprint,
    selecting optimal computational paths based on noise characteristics.
    """
    
    def __init__(self, n_qubits, graph_density=0.3):
        """
        Initialize the DVSP Router.
        
        Args:
            n_qubits: Number of qubits in the system
            graph_density: Density of connections in the braid graph
        """
        self.n_qubits = n_qubits
        self.graph_density = graph_density
        
        # Create the braid graph (representing entanglement pathways)
        self.graph = self._create_braid_graph()
        
        # Noise profile archive (the "Drift Archive" from the blueprint)
        self.noise_profiles = []
        
    def _create_braid_graph(self):
        """Create a graph representing the braided entanglement pathways."""
        # Use NetworkX to create a random graph
        G = nx.erdos_renyi_graph(self.n_qubits, self.graph_density, seed=42)
        
        # Ensure graph is connected (required for pathfinding)
        while not nx.is_connected(G):
            # Add edges until connected
            components = list(nx.connected_components(G))
            if len(components) > 1:
                # Connect first two components
                comp1 = list(components[0])
                comp2 = list(components[1])
                G.add_edge(comp1[0], comp2[0])
            else:
                break
                
        return G
    
    def add_noise_profile(self, profile):
        """
        Add a noise profile to the Drift Archive.
        
        Args:
            profile: Array of noise measurements
        """
        self.noise_profiles.append(profile)
        
        # Keep archive to a reasonable size
        if len(self.noise_profiles) > 100:
            self.noise_profiles.pop(0)
    
    def find_optimal_path(self, start, end):
        """
        Find the optimal computational path using the DVSP algorithm.
        
        Args:
            start: Starting node
            end: Ending node
            
        Returns:
            path: The optimal path as a list of nodes
            metrics: Dictionary of path metrics
        """
        if not self.noise_profiles:
            # No noise profiles yet, use shortest path
            path = nx.shortest_path(self.graph, start, end)
            return path, {"sbq": 0, "riq": 0, "combined": 0}
        
        # Get all simple paths between start and end
        all_paths = list(nx.all_simple_paths(self.graph, start, end))
        
        # If no paths found, return empty path
        if not all_paths:
            return [], {"sbq": 0, "riq": 0, "combined": 0}
        
        # Calculate metrics for each path
        best_path = None
        best_metrics = {"combined": -1}  # Start with worst possible value
        
        for path in all_paths:
            # Calculate SBQ for this path
            sbq_value = sbq(path, self.graph)
            
            # Calculate RIQ using the noise profiles
            # For each node in the path, we use its corresponding noise level
            path_profiles = []
            for node in path:
                # Use node index to extract from profiles, wrapping if needed
                node_profiles = [p[node % len(p)] for p in self.noise_profiles]
                path_profiles.append(node_profiles)
            
            # Flatten profiles for this path
            flat_profiles = [item for sublist in path_profiles for item in sublist]
            riq_value = riq(flat_profiles)
            
            # Combined metric (weighted average)
            combined = combined_metric(sbq_value, riq_value)
            
            # Track best path
            if combined > best_metrics["combined"]:
                best_path = path
                best_metrics = {
                    "sbq": sbq_value,
                    "riq": riq_value,
                    "combined": combined
                }
        
        return best_path, best_metrics


class QDNAmplitudeDampingSimulator:
    """
    Simulator for QDN circuits with amplitude damping noise.
    
    Implements the QDN approach to handling amplitude damping,
    using the DVSP router for adaptive pathfinding.
    """
    
    def __init__(self, n_qubits=5):
        """
        Initialize the simulator.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        
        # Create DVSP router
        self.router = DVSPRouter(n_qubits)
        
        # Initialize simulator
        self.simulator = Aer.get_backend('statevector_simulator')
    
    def create_noise_model(self, damping_rates):
        """
        Create a noise model with amplitude damping.
        
        Args:
            damping_rates: List of damping rates for each qubit
            
        Returns:
            NoiseModel: Qiskit noise model
        """
        noise_model = NoiseModel()
        
        # Add amplitude damping for each qubit with its specific rate
        for i, rate in enumerate(damping_rates):
            error = amplitude_damping_error(rate)
            noise_model.add_quantum_error(error, ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h'], [i])
            
            # Add slightly higher error for two-qubit gates
            if i < self.n_qubits - 1:
                error2 = amplitude_damping_error(rate * 1.2)
                noise_model.add_quantum_error(error2, ['cx'], [i, i+1])
        
        return noise_model
    
    def create_circuit(self, path=None):
        """
        Create a QDN circuit using the specified path.
        
        Args:
            path: Computational path through the qubits (if None, use optimal path)
            
        Returns:
            QuantumCircuit: The created circuit
        """
        # If no path provided, find optimal path from first to last qubit
        if path is None:
            path, _ = self.router.find_optimal_path(0, self.n_qubits - 1)
            
            # If still no path, use sequential path
            if not path:
                path = list(range(self.n_qubits))
        
        # Create circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply Hadamard to first qubit in path
        qc.h(path[0])
        
        # Apply CX gates along the path
        for i in range(len(path) - 1):
            qc.cx(path[i], path[i+1])
        
        # Apply adaptive correction based on the path's characteristics
        # This is the QDN "noise utilization" aspect
        for i, node in enumerate(path):
            # Calculate correction angles based on position in path
            theta = np.pi / (2 * (i + 1))  # Decreases as we move along path
            phi = np.pi / (4 * (len(path) - i))  # Increases as we move along path
            
            # Apply adaptive rotations
            qc.rz(theta, node)
            qc.rx(phi, node)
        
        return qc
    
    def run_simulation(self, damping_rates=None, use_qdn=True):
        """
        Run a simulation with amplitude damping noise.
        
        Args:
            damping_rates: List of damping rates (if None, use random rates)
            use_qdn: Whether to use QDN adaptive pathfinding
            
        Returns:
            dict: Simulation results
        """
        # Generate random damping rates if not provided
        if damping_rates is None:
            damping_rates = np.random.uniform(0.01, 0.1, self.n_qubits)
        
        # Add to noise profiles in the router
        self.router.add_noise_profile(damping_rates)
        
        # Create noise model
        noise_model = self.create_noise_model(damping_rates)
        
        # Create circuits - one with QDN adaptive pathfinding, one without
        if use_qdn:
            # Find optimal path and create circuit
            path, metrics = self.router.find_optimal_path(0, self.n_qubits - 1)
            qc = self.create_circuit(path)
        else:
            # Standard sequential circuit
            qc = self.create_circuit(list(range(self.n_qubits)))
            metrics = {"sbq": 0, "riq": 0, "combined": 0}
        
        # Transpile and run
        t_qc = transpile(qc, self.simulator)
        result = self.simulator.run(t_qc, noise_model=noise_model).result()
        
        # Get statevector
        state = result.get_statevector()
        
        # Calculate fidelity to ideal GHZ-like state
        # For GHZ-like state, we expect superposition of |00...0⟩ and |11...1⟩
        ideal_state = Statevector.from_label('0' * self.n_qubits)
        ideal_state = ideal_state.evolve(QuantumCircuit(self.n_qubits).h(0).cx(0, range(1, self.n_qubits)))
        
        fidelity = state_fidelity(state, ideal_state)
        
        return {
            "fidelity": fidelity,
            "metrics": metrics,
            "damping_rates": damping_rates,
            "circuit": qc,
            "state": state
        }
    
    def compare_approaches(self, n_trials=10):
        """
        Compare standard and QDN approaches across multiple trials.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            dict: Comparison results
        """
        standard_fidelities = []
        qdn_fidelities = []
        improvements = []
        
        for i in range(n_trials):
            # Generate random damping rates for this trial
            damping_rates = np.random.uniform(0.01, 0.1, self.n_qubits)
            
            # Run both approaches with the same noise
            standard_result = self.run_simulation(damping_rates, use_qdn=False)
            qdn_result = self.run_simulation(damping_rates, use_qdn=True)
            
            # Record fidelities
            standard_fidelity = standard_result["fidelity"]
            qdn_fidelity = qdn_result["fidelity"]
            
            standard_fidelities.append(standard_fidelity)
            qdn_fidelities.append(qdn_fidelity)
            
            # Calculate improvement
            improvement = ((qdn_fidelity - standard_fidelity) / standard_fidelity) * 100
            improvements.append(improvement)
            
            print(f"Trial {i+1}/{n_trials}:")
            print(f"  Standard Fidelity: {standard_fidelity:.4f}")
            print(f"  QDN Fidelity:      {qdn_fidelity:.4f}")
            print(f"  Improvement:       {improvement:.2f}%")
            print(f"  QDN Metrics: SBQ={qdn_result['metrics']['sbq']:.2f}, "
                  f"RIQ={qdn_result['metrics']['riq']:.2f}, "
                  f"Combined={qdn_result['metrics']['combined']:.2f}")
            print()
        
        # Calculate averages
        avg_standard = np.mean(standard_fidelities)
        avg_qdn = np.mean(qdn_fidelities)
        avg_improvement = np.mean(improvements)
        
        return {
            "standard_fidelities": standard_fidelities,
            "qdn_fidelities": qdn_fidelities,
            "improvements": improvements,
            "avg_standard": avg_standard,
            "avg_qdn": avg_qdn,
            "avg_improvement": avg_improvement
        }
    
    def plot_comparison(self, results):
        """
        Plot comparison results.
        
        Args:
            results: Results from compare_approaches
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot fidelities
        trials = range(1, len(results["standard_fidelities"]) + 1)
        ax1.plot(trials, results["standard_fidelities"], 'o-', label='Standard')
        ax1.plot(trials, results["qdn_fidelities"], 's-', label='QDN')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Fidelity')
        ax1.set_title('Fidelity Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot improvements
        ax2.bar(trials, results["improvements"])
        ax2.axhline(y=results["avg_improvement"], color='r', linestyle='-', 
                   label=f'Avg: {results["avg_improvement"]:.2f}%')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('QDN Improvement over Standard')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig


def visualize_braid_graph(router):
    """
    Visualize the braid graph with the optimal path highlighted.
    
    Args:
        router: DVSPRouter instance
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Find optimal path
    path, metrics = router.find_optimal_path(0, router.n_qubits - 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Position nodes in a circular layout
    pos = nx.circular_layout(router.graph)
    
    # Draw the graph
    nx.draw_networkx_nodes(router.graph, pos, node_color='lightblue', 
                           node_size=500, ax=ax)
    nx.draw_networkx_edges(router.graph, pos, width=1, alpha=0.5, ax=ax)
    
    # Highlight the path
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        nx.draw_networkx_nodes(router.graph, pos, nodelist=path, 
                               node_color='orange', node_size=500, ax=ax)
        nx.draw_networkx_edges(router.graph, pos, edgelist=path_edges, 
                               width=3, edge_color='orange', ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(router.graph, pos, ax=ax)
    
    # Add metrics
    metrics_text = f"SBQ: {metrics['sbq']:.2f}\nRIQ: {metrics['riq']:.2f}\nCombined: {metrics['combined']:.2f}"
    plt.figtext(0.02, 0.02, metrics_text, fontsize=12)
    
    plt.title("Braid Graph with Optimal Path")
    plt.axis('off')
    
    return fig


def main():
    """Run a demonstration of QDN amplitude damping simulation."""
    print("Quantum Drift Nexus - Amplitude Damping Simulation")
    print("=" * 50)
    
    # Create simulator
    n_qubits = 7  # Phase II target: 7-10 qubits
    simulator = QDNAmplitudeDampingSimulator(n_qubits=n_qubits)
    
    print(f"Created simulator with {n_qubits} qubits")
    print()
    
    # Visualize the braid graph
    print("Visualizing braid graph...")
    fig = visualize_braid_graph(simulator.router)
    plt.savefig('braid_graph.png')
    plt.close(fig)
    print("Saved braid graph visualization to 'braid_graph.png'")
    print()
    
    # Run comparison
    print("Comparing standard vs QDN approaches...")
    results = simulator.compare_approaches(n_trials=5)
    
    print("\nSummary:")
    print(f"Average Standard Fidelity: {results['avg_standard']:.4f}")
    print(f"Average QDN Fidelity:      {results['avg_qdn']:.4f}")
    print(f"Average Improvement:       {results['avg_improvement']:.2f}%")
    
    # Plot comparison
    fig = simulator.plot_comparison(results)
    plt.savefig('amplitude_damping_comparison.png')
    plt.close(fig)
    print("Saved comparison plot to 'amplitude_damping_comparison.png'")


if __name__ == "__main__":
    main()
