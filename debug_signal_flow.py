"""Debug script for understanding signal flow."""

import torch
from pyFDN.recursive import (
    DelayRead, DelayWrite, FeedbackMix, InputTap, OutputTap, RecursionCore
)

# Simple feedback comb filter
delay_length = 4
feedback_gain = 0.7

stages = [
    DelayRead(delay_length=delay_length, num_lines=1),
    FeedbackMix(feedback_matrix=torch.tensor([[feedback_gain]])),
    InputTap(input_matrix=torch.ones(1, 1)),
    DelayWrite(),
    OutputTap(num_lines=1, num_outputs=1),
]

core = RecursionCore(stages)

# Manually step through to see state
state = core.init_state(batch_size=1)
print("Initial buffer shape:", state["delay_buffers"].shape)
print("Initial buffer:", state["delay_buffers"][0, :, 0])
print("Initial pointer:", state["delay_pointer"][0])

# Impulse input
input_signal = torch.zeros(20, 1)
input_signal[0, 0] = 1.0

output = core.process(input_signal, block_size=2)  # Use small block size!

print("Input:")
print(input_signal[:12].T)
print("\nOutput:")
print(output[:12].T)
print("\nExpected peaks at samples:", [0, 4, 8])
print("Expected values:", [1.0, 0.7, 0.49])
print("\nActual values at those samples:")
for idx in [0, 4, 8]:
    print(f"  output[{idx}] = {output[idx, 0].item():.4f}")
