"""Core coordinator for recursive DSP processing."""

from __future__ import annotations
from typing import Dict, List, Optional
import torch

from .stage import Stage


class RecursionCore:
    """
    Coordinator for recursive DSP processing using a sequence of stages.
    
    This class manages:
    - Ordered list of processing stages
    - Global state dictionary shared across all stages
    - Block-based processing of input signals
    - Context (ctx) propagation through stages
    
    Typical usage:
        stages = [DelayRead(...), FeedbackMix(...), InputTap(...), 
                  DelayWrite(...), OutputTap(...)]
        core = RecursionCore(stages)
        output = core.process(input_signal, block_size=512)
    """
    
    def __init__(self, stages: List[Stage], device: Optional[torch.device] = None):
        """
        Initialize the recursion core.
        
        Args:
            stages: Ordered list of Stage instances to execute sequentially
            device: PyTorch device for all tensors. If None, uses CPU.
        """
        self.stages = stages
        self.device = device or torch.device("cpu")
        self._validate_stages()
        
    def _validate_stages(self) -> None:
        """
        Validate stage configuration.
        
        Note: Multiple stages can claim the same state keys (e.g., DelayRead and
        DelayWrite both use delay_buffers), but only one should initialize them.
        We detect conflicts during init_state if multiple stages try to initialize
        the same key.
        """
        # Just track state keys for informational purposes
        # Actual conflicts will be caught in init_state
        pass
    
    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Initialize global state by merging all stage states.
        
        Args:
            batch_size: Number of parallel signals (B dimension)
            
        Returns:
            Merged state dictionary containing all stage states
        """
        state: Dict[str, torch.Tensor] = {}
        for stage in self.stages:
            stage_state = stage.init_state(batch_size, self.device)
            # Ensure no key conflicts (should already be caught by validation)
            for key in stage_state:
                if key in state:
                    raise RuntimeError(
                        f"State key '{key}' initialized by multiple stages"
                    )
                state[key] = stage_state[key]
        return state
    
    def process(
        self,
        input_signal: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
        """
        Process input signal through the recursive system.
        
        Args:
            input_signal: Input tensor of shape [T_total, N_in] or [B, T_total, N_in]
            block_size: Maximum number of samples per processing block
            
        Returns:
            Output tensor of same shape as input (with N_out channels instead of N_in)
            
        Note:
            - Automatically adds batch dimension if input is 2D
            - Handles variable-length final block if T_total % block_size != 0
            - Maintains device and dtype consistency
        """
        # Ensure input is 3D [B, T_total, N_in]
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        elif input_signal.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(
                f"Input must be 2D [T, N_in] or 3D [B, T, N_in], "
                f"got {input_signal.dim()}D"
            )
        
        # Move to correct device
        input_signal = input_signal.to(self.device)
        
        B, T_total, N_in = input_signal.shape
        
        # Initialize state
        state = self.init_state(B)
        
        # Determine output shape (from last OutputTap stage or default to N_in)
        N_out = N_in  # Will be updated by first block output
        
        # Prepare output buffer (will be allocated on first block)
        output_blocks: List[torch.Tensor] = []
        
        # Process blocks
        num_blocks = (T_total + block_size - 1) // block_size
        
        for block_idx in range(num_blocks):
            # Determine current block size
            start_t = block_idx * block_size
            end_t = min(start_t + block_size, T_total)
            current_block_size = end_t - start_t
            
            # Extract input block
            x_block = input_signal[:, start_t:end_t, :]  # [B, T, N_in]
            
            # Initialize context for this block
            ctx: Dict[str, torch.Tensor] = {
                "x": x_block,
            }
            
            # Initialize next_state accumulator
            next_state: Dict[str, torch.Tensor] = {}
            
            # Run all stages in sequence
            for stage in self.stages:
                stage.step_block(ctx, state, next_state, current_block_size)
            
            # Merge state_t and next_state for next block
            for key in state:
                if key in next_state:
                    state[key] = next_state[key]
                # else: carry over previous state value
            
            # Extract output block
            if "y" not in ctx:
                raise RuntimeError(
                    "No output produced - ensure pipeline includes OutputTap stage"
                )
            
            y_block = ctx["y"]  # [B, T, N_out]
            output_blocks.append(y_block)
        
        # Concatenate all output blocks
        output = torch.cat(output_blocks, dim=1)  # [B, T_total, N_out]
        
        # Remove batch dimension if input was 2D
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def __repr__(self) -> str:
        stage_list = "\n  ".join(str(s) for s in self.stages)
        return f"RecursionCore(\n  {stage_list}\n)"
