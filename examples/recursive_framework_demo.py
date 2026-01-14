"""
Recursive DSP Framework Demonstration
======================================

This script demonstrates the usage of the recursive DSP framework
with various example systems.
"""

import torch
import matplotlib.pyplot as plt
from pyFDN.recursive import (
    DelayRead, DelayWrite, Biquads,
    FeedbackMix, InputTap, OutputTap, RecursionCore
)


def create_impulse(length: int, num_channels: int = 1) -> torch.Tensor:
    """Create an impulse signal."""
    signal = torch.zeros(length, num_channels)
    signal[0, :] = 1.0
    return signal


def plot_signal(signal: torch.Tensor, title: str, max_samples: int = 500):
    """Plot a signal."""
    plt.figure(figsize=(12, 4))
    signal_np = signal.squeeze().numpy()
    plt.plot(signal_np[:max_samples])
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def example_1_pure_delay():
    """Example 1: Simple pure delay (no feedback)."""
    print("\n" + "="*60)
    print("Example 1: Pure Delay System")
    print("="*60)
    
    delay_length = 100
    num_lines = 1
    
    stages = [
        DelayRead(delay_length=delay_length, num_lines=num_lines),
        DelayWrite(),
        OutputTap(num_lines=num_lines, num_outputs=num_lines),
    ]
    
    core = RecursionCore(stages)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(300, num_lines)
    output = core.process(input_signal, block_size=64)
    
    print(f"\nInput shape: {input_signal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Delay length: {delay_length} samples")
    
    plot_signal(output, f"Pure Delay - {delay_length} samples")
    return output


def example_2_feedback_comb():
    """Example 2: Feedback comb filter."""
    print("\n" + "="*60)
    print("Example 2: Feedback Comb Filter")
    print("="*60)
    
    delay_length = 50
    feedback_gain = 0.7
    
    stages = [
        DelayRead(delay_length=delay_length, num_lines=1),
        FeedbackMix(feedback_matrix=torch.tensor([[feedback_gain]])),
        InputTap(input_matrix=torch.ones(1, 1)),
        DelayWrite(),
        OutputTap(num_lines=1, num_outputs=1),
    ]
    
    core = RecursionCore(stages)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(500, 1)
    output = core.process(input_signal, block_size=64)
    
    print(f"\nDelay: {delay_length} samples")
    print(f"Feedback gain: {feedback_gain}")
    print(f"Expected periodicity: {delay_length} samples")
    
    plot_signal(output, f"Feedback Comb Filter (delay={delay_length}, g={feedback_gain})")
    return output


def example_3_fdn_absorption_inside():
    """Example 3: FDN with absorption inside feedback loop."""
    print("\n" + "="*60)
    print("Example 3: FDN with Absorption Inside Loop")
    print("="*60)
    
    num_lines = 4
    delay_length = 64
    
    # Create Hadamard feedback matrix
    A = torch.tensor([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ], dtype=torch.float32) * 0.5
    
    # Lowpass absorption filters (one-pole, a=0.85)
    # Format: [a0, a1, a2, b0, b1, b2] where a0=1.0, a1=-0.85, b0=0.15
    absorption_coeffs = torch.tensor([[[1.0, -0.85, 0.0, 0.15, 0.0, 0.0]]]).repeat(num_lines, 1, 1)
    
    stages = [
        DelayRead(delay_length=delay_length, num_lines=num_lines),
        FeedbackMix(feedback_matrix=A),
        Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
        InputTap(input_matrix=torch.ones(num_lines, 1)),
        DelayWrite(),
        OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
    ]
    
    core = RecursionCore(stages)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(1000, 1)
    output = core.process(input_signal, block_size=128)
    
    print(f"\nNumber of delay lines: {num_lines}")
    print(f"Delay length: {delay_length} samples")
    print(f"Feedback matrix: Hadamard (orthogonal)")
    print(f"Absorption: One-pole lowpass (a=0.85)")
    
    plot_signal(output, "FDN with Absorption Inside Loop", max_samples=1000)
    return output


def example_4_fdn_absorption_outside():
    """Example 4: FDN with absorption outside feedback loop."""
    print("\n" + "="*60)
    print("Example 4: FDN with Absorption Outside Loop")
    print("="*60)
    
    num_lines = 4
    delay_length = 64
    
    # Diagonal feedback matrix with gain
    A = torch.eye(num_lines) * 0.9
    
    # Lowpass absorption filters (one-pole, a=0.85)
    # Format: [a0, a1, a2, b0, b1, b2] where a0=1.0, a1=-0.85, b0=0.15
    absorption_coeffs = torch.tensor([[[1.0, -0.85, 0.0, 0.15, 0.0, 0.0]]]).repeat(num_lines, 1, 1)
    
    stages = [
        DelayRead(delay_length=delay_length, num_lines=num_lines),
        FeedbackMix(feedback_matrix=A),
        InputTap(input_matrix=torch.ones(num_lines, 1)),
        DelayWrite(),
        Biquads(num_lines=num_lines, biquad_coeffs=absorption_coeffs),
        OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
    ]
    
    core = RecursionCore(stages)
    print(f"\nSystem structure:\n{core}")
    
    # Process impulse
    input_signal = create_impulse(1000, 1)
    output = core.process(input_signal, block_size=128)
    
    print(f"\nNumber of delay lines: {num_lines}")
    print(f"Delay length: {delay_length} samples")
    print(f"Feedback matrix: Diagonal (g=0.9)")
    print(f"Absorption: One-pole lowpass (a=0.85) - OUTSIDE loop")
    
    plot_signal(output, "FDN with Absorption Outside Loop", max_samples=1000)
    return output


def example_5_block_size_comparison():
    """Example 5: Verify block size invariance."""
    print("\n" + "="*60)
    print("Example 5: Block Size Invariance")
    print("="*60)
    
    num_lines = 2
    delay_length = 32
    
    stages_template = lambda: [
        DelayRead(delay_length=delay_length, num_lines=num_lines),
        FeedbackMix(feedback_matrix=torch.eye(num_lines) * 0.7),
        InputTap(input_matrix=torch.ones(num_lines, 1)),
        DelayWrite(),
        OutputTap(output_matrix=torch.ones(1, num_lines) / num_lines),
    ]
    
    input_signal = torch.randn(256, 1)
    
    block_sizes = [8, 32, 64, 256]
    outputs = {}
    
    for bs in block_sizes:
        core = RecursionCore(stages_template())
        output = core.process(input_signal.clone(), block_size=bs)
        outputs[bs] = output
        print(f"\nBlock size {bs:3d}: Output shape {output.shape}")
    
    # Compare outputs
    print("\nComparing outputs (max absolute difference):")
    reference = outputs[block_sizes[0]]
    for bs in block_sizes[1:]:
        diff = (outputs[bs] - reference).abs().max().item()
        print(f"  Block size {block_sizes[0]} vs {bs}: {diff:.2e}")
    
    return outputs


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RECURSIVE DSP FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Run examples
    try:
        example_1_pure_delay()
        example_2_feedback_comb()
        example_3_fdn_absorption_inside()
        example_4_fdn_absorption_outside()
        example_5_block_size_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
        # Show all plots
        plt.show()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
