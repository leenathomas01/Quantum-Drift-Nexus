"""
Quantum Drift Nexus (QDN) - Reinforcement Learning Based Adaptation

This module demonstrates the bio-mimetic adaptability of QDN through a simplified
reinforcement learning approach that optimizes quantum circuit parameters in response
to environmental noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit.visualization import plot_histogram


class QDNEnvironment:
    """A simplified environment for QDN reinforcement learning experiments."""
    
    def __init__(self, num_qubits=2, base_noise_level=0.05, noise_volatility=0.02):
        """Initialize the QDN environment.
        
        Args:
            num_qubits: Number of qubits in the base circuit
            base_noise_level: Starting noise level
            noise_volatility: How much the noise can fluctuate
        """
        self.num_qubits = num_qubits
        self.base_noise_level = base_noise_level
        self.noise_volatility = noise_volatility
        self.current_noise_level = base_noise_level
        self.simulator = Aer.get_backend('qasm_simulator')
        self.shots = 1024
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Randomize the current noise level around the base
        self.current_noise_level = max(0.01, 
                                       self.base_noise_level + 
                                       np.random.normal(0, self.noise_volatility))
        
        # Create a new noise model
        self.noise_model = self._create_noise_model(self.current_noise_level)
        
        # Return initial observation
        return {'noise_level': self.current_noise_level}
    
    def _create_noise_model(self, noise_level):
        """Create a noise model with the given noise level."""
        noise_model = NoiseModel()
        
        # Depolarizing error
        error_1q = depolarizing_error(noise_level, 1)
        error_2q = depolarizing_error(noise_level, 2)
        
        # Amplitude damping
        error_amp = amplitude_damping_error(noise_level * 1.5)
        
        # Add errors to the noise model
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        noise_model.add_all_qubit_quantum_error(error_amp, ['u1', 'u2', 'u3', 'h', 'rx', 'ry', 'rz'])
        
        return noise_model
    
    def step(self, action):
        """Take an action in the environment and return the result.
        
        Args:
            action: Dictionary of circuit parameters
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Extract action parameters
        redundancy = int(max(1, action.get('redundancy', 1)))
        theta_factor = action.get('theta_factor', 1.0)
        phi_factor = action.get('phi_factor', 1.0)
        
        # Create circuit with given parameters
        circuit = self._create_qdn_circuit(
            redundancy=redundancy,
            theta_factor=theta_factor,
            phi_factor=phi_factor
        )
        
        # Run the circuit with current noise model
        transpiled_circuit = transpile(circuit, self.simulator)
        result = self.simulator.run(
            transpiled_circuit,
            noise_model=self.noise_model,
            shots=self.shots
        ).result()
        
        # Get counts and calculate fidelity
        counts = result.get_counts()
        fidelity = self._calculate_bell_fidelity(counts)
        
        # Calculate reward based on fidelity
        # We want high fidelity but also efficiency (lower redundancy is better if fidelity is good)
        efficiency_penalty = 0.05 * (redundancy - 1)  # Penalty for using more qubits
        reward = fidelity - efficiency_penalty
        
        # Update noise level for next step (simulating environmental changes)
        self._update_noise_level()
        
        # Return observation, reward, done, info
        observation = {
            'noise_level': self.current_noise_level,
            'last_fidelity': fidelity
        }
        
        done = False  # In a real RL setup, we'd have episode termination conditions
        info = {
            'counts': counts,
            'fidelity': fidelity,
            'efficiency': 1.0 - efficiency_penalty
        }
        
        return observation, reward, done, info
    
    def _create_qdn_circuit(self, redundancy=1, theta_factor=1.0, phi_factor=1.0):
        """Create a QDN circuit with the given parameters."""
        # Total qubits with redundancy
        total_qubits = self.num_qubits * redundancy
        
        # Create the circuit
        qc = QuantumCircuit(total_qubits, self.num_qubits)
        
        # Apply redundant encoding
        for r in range(redundancy):
            # Base index for this redundant copy
            base = r * self.num_qubits
            
            # Create Bell state
            qc.h(base)
            qc.cx(base, base + 1)
            
            # Apply adaptive correction based on estimated noise and parameters
            theta = np.pi/2 * (1 - self.current_noise_level * theta_factor)
            phi = np.pi/4 * self.current_noise_level * phi_factor
            
            # Adaptive rotation gates
            qc.rz(theta, base)
            qc.rx(phi, base + 1)
        
        # Apply holographic encoding (connections between redundant copies)
        if redundancy > 1:
            for r in range(redundancy-1):
                qc.cx(r * self.num_qubits, (r+1) * self.num_qubits)
        
        # Measurement
        for i in range(self.num_qubits):
            qc.measure(i, i)
            
        return qc
    
    def _calculate_bell_fidelity(self, counts):
        """Calculate the fidelity of measurement results to an ideal Bell state."""
        # For a Bell state, we should see only '00' and '11' with equal probability
        total_shots = sum(counts.values())
        correct_shots = counts.get('00', 0) + counts.get('11', 0)
        
        # Simple fidelity metric
        fidelity = correct_shots / total_shots
        return fidelity
    
    def _update_noise_level(self):
        """Update the noise level to simulate changing environment."""
        # Random walk the noise level
        noise_delta = np.random.normal(0, self.noise_volatility)
        self.current_noise_level = max(0.01, min(0.3, self.current_noise_level + noise_delta))
        
        # Update the noise model
        self.noise_model = self._create_noise_model(self.current_noise_level)
        
    def visualize_circuit(self, action):
        """Visualize the circuit for a given action."""
        circuit = self._create_qdn_circuit(
            redundancy=int(max(1, action.get('redundancy', 1))),
            theta_factor=action.get('theta_factor', 1.0),
            phi_factor=action.get('phi_factor', 1.0)
        )
        return circuit.draw(output='mpl')


class QDNAgent:
    """A simple reinforcement learning agent for QDN parameter optimization."""
    
    def __init__(self, action_space):
        """Initialize the agent.
        
        Args:
            action_space: Dictionary of parameter ranges {param: (min, max)}
        """
        self.action_space = action_space
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.exploration_decay = 0.95
        self.discount_factor = 0.9
        self.parameters = {param: (min_val + max_val) / 2 for param, (min_val, max_val) in action_space.items()}
        self.best_parameters = self.parameters.copy()
        self.best_reward = -float('inf')
        self.history = []
    
    def get_action(self, observation=None):
        """Choose an action based on the current policy and observation.
        
        Args:
            observation: Current environment observation (not used in this simple agent)
            
        Returns:
            Dictionary of action parameters
        """
        # Either explore or exploit
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            action = {}
            for param, (min_val, max_val) in self.action_space.items():
                if param == 'redundancy':  # Integer parameter
                    action[param] = np.random.randint(min_val, max_val + 1)
                else:  # Continuous parameter
                    action[param] = np.random.uniform(min_val, max_val)
        else:
            # Exploitation: use current parameters with small random perturbations
            action = {}
            for param, value in self.parameters.items():
                min_val, max_val = self.action_space[param]
                if param == 'redundancy':  # Integer parameter
                    action[param] = max(min_val, min(max_val, 
                                                   int(value + np.random.randint(-1, 2))))
                else:  # Continuous parameter
                    noise = 0.1 * (max_val - min_val)
                    action[param] = max(min_val, min(max_val, 
                                                    value + np.random.normal(0, noise)))
        
        return action
    
    def update(self, observation, action, reward, next_observation):
        """Update the agent's parameters based on the observed reward.
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
        """
        # Very simple update rule: move parameters toward better-performing actions
        for param, value in action.items():
            # Update rate depends on how good the reward was
            update_rate = self.learning_rate * max(0, reward)
            self.parameters[param] = (1 - update_rate) * self.parameters[param] + update_rate * value
        
        # Track best parameters
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_parameters = action.copy()
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        
        # Record history
        self.history.append({
            'action': action.copy(),
            'reward': reward,
            'parameters': self.parameters.copy()
        })
    
    def get_best_action(self):
        """Return the best action found so far."""
        return self.best_parameters


def bcm(counts):
    """Calculate Braid Coherence Metric (BCM) - a measure of entanglement resilience.
    
    Args:
        counts: Measurement counts from a quantum circuit
        
    Returns:
        Float value representing the BCM
    """
    # In a real implementation, this would analyze entanglement structure
    # For this simulation, we use a simplified metric based on state purity
    total_shots = sum(counts.values())
    
    # Calculate a simplified "purity" as our BCM
    purity = sum((count/total_shots)**2 for count in counts.values())
    
    # Scale between 0-1 where higher is better
    # Ideal Bell state would have purity of 0.5
    bcm = 2 * purity if purity <= 0.5 else 2 * (1 - purity)
    return bcm


def run_qdn_rl_experiment(episodes=50, steps_per_episode=10):
    """Run a reinforcement learning experiment to optimize QDN parameters.
    
    Args:
        episodes: Number of training episodes
        steps_per_episode: Number of steps per episode
        
    Returns:
        Agent with optimized parameters
    """
    # Define the action space
    action_space = {
        'redundancy': (1, 3),      # Number of redundant copies (integer)
        'theta_factor': (0.5, 2.0), # Factor for theta angle calculation
        'phi_factor': (0.5, 2.0)    # Factor for phi angle calculation
    }
    
    # Create environment and agent
    env = QDNEnvironment(num_qubits=2, base_noise_level=0.05)
    agent = QDNAgent(action_space)
    
    # Track results
    episode_rewards = []
    episode_fidelities = []
    noise_levels = []
    
    # Training loop
    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0
        episode_fidelity = []
        
        print(f"Episode {episode+1}/{episodes} - Starting noise: {observation['noise_level']:.4f}")
        
        for step in range(steps_per_episode):
            # Choose action
            action = agent.get_action(observation)
            
            # Take step in environment
            next_observation, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(observation, action, reward, next_observation)
            
            # Track metrics
            total_reward += reward
            episode_fidelity.append(info['fidelity'])
            noise_levels.append(observation['noise_level'])
            
            # Print progress
            print(f"  Step {step+1}: Noise={observation['noise_level']:.4f}, " + 
                  f"Action={action}, Reward={reward:.4f}, Fidelity={info['fidelity']:.4f}")
            
            # Update observation
            observation = next_observation
            
            if done:
                break
        
        # Track episode results
        episode_rewards.append(total_reward / steps_per_episode)
        episode_fidelities.append(np.mean(episode_fidelity))
        
        print(f"Episode {episode+1} complete - Avg reward: {episode_rewards[-1]:.4f}, " + 
              f"Avg fidelity: {episode_fidelities[-1]:.4f}")
        print(f"Best parameters so far: {agent.best_parameters}")
        print()
    
    # Plot learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Reward curve
    ax1.plot(range(1, episodes+1), episode_rewards, 'b-')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curve - Reward')
    ax1.grid(alpha=0.3)
    
    # Fidelity curve
    ax2.plot(range(1, episodes+1), episode_fidelities, 'g-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Fidelity')
    ax2.set_title('Learning Curve - Fidelity')
    ax2.grid(alpha=0.3)
    
    # Noise levels
    ax3.plot(range(1, len(noise_levels)+1), noise_levels, 'r-', alpha=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Noise Level')
    ax3.set_title('Environmental Noise Variation')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qdn_rl_learning.png')
    plt.show()
    
    # Test best parameters
    print("\nTesting best parameters:")
    env.reset()
    
    # Standard circuit (no QDN)
    standard_circuit = QuantumCircuit(2, 2)
    standard_circuit.h(0)
    standard_circuit.cx(0, 1)
    standard_circuit.measure([0, 1], [0, 1])
    
    standard_result = env.simulator.run(
        transpile(standard_circuit, env.simulator),
        noise_model=env.noise_model,
        shots=1024
    ).result()
    standard_counts = standard_result.get_counts()
    
    # QDN circuit with best parameters
    best_action = agent.get_best_action()
    observation, reward, _, info = env.step(best_action)
    
    # Compare results
    qdn_counts = info['counts']
    standard_fidelity = env._calculate_bell_fidelity(standard_counts)
    qdn_fidelity = info['fidelity']
    
    print(f"Noise level: {env.current_noise_level:.4f}")
    print(f"Standard circuit fidelity: {standard_fidelity:.4f}")
    print(f"QDN circuit fidelity: {qdn_fidelity:.4f}")
    print(f"Improvement: {(qdn_fidelity - standard_fidelity) / standard_fidelity * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plot_histogram([standard_counts, qdn_counts], 
                   legend=['Standard Circuit', 'QDN Optimized'],
                   title=f"Comparison (Noise: {env.current_noise_level:.4f})")
    plt.savefig('qdn_rl_comparison.png')
    plt.show()
    
    return agent


if __name__ == "__main__":
    # Run a smaller experiment for demonstration
    agent = run_qdn_rl_experiment(episodes=10, steps_per_episode=5)
    
    print("\nOptimized QDN Parameters:")
    for param, value in agent.best_parameters.items():
        print(f"  {param}: {value:.4f}" if param != 'redundancy' else f"  {param}: {int(value)}")
