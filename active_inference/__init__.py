"""Active inference engine — perception, learning, and action via free energy minimization."""

from active_inference.generative_model import GenerativeModel
from active_inference.agent import ActiveInferenceAgent
from active_inference.free_energy import variational_free_energy, expected_free_energy

__all__ = [
    "GenerativeModel",
    "ActiveInferenceAgent",
    "variational_free_energy",
    "expected_free_energy",
]
