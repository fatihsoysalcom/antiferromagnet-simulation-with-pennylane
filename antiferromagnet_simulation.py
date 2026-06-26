import pennylane as qml
from pennylane import numpy as np

# Define the number of qubits (spins)
num_spins = 2

# Define the Hamiltonian for a simple antiferromagnet (e.g., Ising model with alternating couplings)
# For two spins, this can be represented as J * S_z(1) * S_z(2)
# In PennyLane, this translates to J * qml.PauliZ(0) @ qml.PauliZ(1)
# We'll use a simplified representation for demonstration.

def hamiltonian(J):
    # For two spins, a simple antiferromagnetic interaction is -J * Z_0 @ Z_1 where J > 0
    # This encourages opposite spins.
    return -J * qml.PauliZ(0) @ qml.PauliZ(1)

# Define a quantum device
device = qml.device("default.qubit", wires=num_spins)

# Define the variational circuit (ansatz)
def ansatz(params):
    # A simple ansatz: apply RY rotations and CNOTs
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])

# Define the cost function (expectation value of the Hamiltonian)
@qml.qnode(device)
def cost_fn(params, J):
    ansatz(params)
    # Return the expectation value of the defined Hamiltonian
    return qml.expval(hamiltonian(J))

# --- Simulation Parameters ---
J_value = 1.0  # Antiferromagnetic coupling strength

# Initialize variational parameters
# We need as many parameters as there are trainable gates in the ansatz
initial_params = np.array([0.1, 0.1], requires_grad=True)

# Define an optimizer
optimizer = qml.AdamOptimizer(stepsize=0.1)

# --- Optimization Loop ---
params = initial_params
max_iterations = 100

print(f"Starting optimization for J = {J_value}")

for n in range(max_iterations):
    # Update parameters using the optimizer
    params, cost = optimizer.step_and_cost(lambda p: cost_fn(p, J=J_value), params)

    if (n + 1) % 10 == 0:
        print(f"Iteration {n+1}: Cost = {cost:.4f}")

print("Optimization finished.")
print(f"Optimized parameters: {params}")

# --- Final State and Expectation Values ---
# Run the circuit with optimized parameters to get the final state
# (This part is illustrative, cost_fn already computed the expectation value)
final_expectation_value = cost_fn(params, J=J_value)
print(f"Final expectation value of the Hamiltonian: {final_expectation_value:.4f}")

# To see the ground state, we can analyze the state vector
# Re-run the circuit and get the state vector
@qml.qnode(device)
def get_state(params):
    ansatz(params)
    return qml.state()

final_state = get_state(params)
print(f"Final quantum state vector: {final_state}")

# For a true antiferromagnet, we expect spins to be anti-aligned.
# For the Ising model with Z_0 @ Z_1, the ground states are |01> and |10>.
# The expectation value of Z_0 @ Z_1 should be -1 for these states.
# Our Hamiltonian is -J * Z_0 @ Z_1, so the ground state energy is J.
# The expectation value of the Hamiltonian should be close to J.

# Let's check the expectation values of individual spins
exp_z0 = qml.expval(qml.PauliZ(0))
exp_z1 = qml.expval(qml.PauliZ(1))

# We need to re-run the qnode to get these expectation values with the optimized params
@qml.qnode(device)
def get_individual_expvals(params):
    ansatz(params)
    exp_z0 = qml.expval(qml.PauliZ(0))
    exp_z1 = qml.expval(qml.PauliZ(1))
    return exp_z0, exp_z1

exp_z0_final, exp_z1_final = get_individual_expvals(params)
print(f"Expectation value of PauliZ on spin 0: {exp_z0_final:.4f}")
print(f"Expectation value of PauliZ on spin 1: {exp_z1_final:.4f}")

# Ideally, for an antiferromagnetic ground state, exp_z0 should be close to +1 and exp_z1 close to -1 (or vice versa).
# This indicates anti-alignment of spins.
