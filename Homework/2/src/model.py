"""
Conv-TasNet Model Implementation for Audio Source Separation.

This module implements the Conv-TasNet (Convolutional Time-domain Audio Separation Network)
architecture for monaural audio source separation. The model consists of an encoder,
separator network (TCN), and decoder components.

Classes:
    Encoder: 1D convolutional encoder for audio mixture representation
    Decoder: 1D transposed convolutional decoder for source reconstruction
    GlobalLayerNorm: Global layer normalization implementation
    ChannelwiseLayerNorm: Channel-wise layer normalization implementation
    DepthwiseSeparableConv1d: Depthwise separable convolution block
    TCNBlock: Temporal Convolutional Network block with residual connections
    TCN: Full TCN separator network with multiple blocks and repeats
    ConvTasNet: Complete Conv-TasNet model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import cast, Iterable

# Import config from the same directory
import config


# --- Model Components ---
class Encoder(nn.Module):
    """
    1D Convolutional encoder for audio mixture representation.

    Converts time-domain audio mixture into encoded representation using
    1D convolution followed by ReLU activation.

    Args:
        l (int): Kernel size for encoder convolution
        n (int): Number of encoder filters (output channels)
    """

    def __init__(self, l=config.L_CONV_KERNEL_SIZE, n=config.N_ENCODER_FILTERS):
        super(Encoder, self).__init__()
        self.conv1d_U = nn.Conv1d(
            1, n, kernel_size=l, stride=l // 2, padding=0, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, mixture):  # mixture: (batch, 1, T_samples)
        """Forward pass through encoder."""
        mixture_w = self.relu(self.conv1d_U(mixture))  # (batch, n, T_frames)
        return mixture_w


class Decoder(nn.Module):
    """
    1D Transposed convolutional decoder for source reconstruction.

    Converts encoded source representations back to time-domain audio signals
    using transposed convolution with proper length matching.

    Args:
        n (int): Number of input channels (encoder filters)
        l (int): Kernel size for decoder convolution
    """

    def __init__(self, n=config.N_ENCODER_FILTERS, l=config.L_CONV_KERNEL_SIZE):
        super(Decoder, self).__init__()
        self.N = n
        self.L = l
        self.conv_transpose1d_V = nn.ConvTranspose1d(
            n, 1, kernel_size=l, stride=l // 2, bias=False
        )

    def forward(
        self, source_w_masked, original_mixture_len
    ):  # source_w_masked: (batch, n, T_frames)
        """
        Forward pass through decoder with length matching.

        Args:
            source_w_masked: Encoded and masked source representation (batch, n, T_frames)
            original_mixture_len: Target length for output signal

        Returns:
            est_source: Reconstructed time-domain source (batch_size*n_sources, 1, T_samples)
        """
        est_source = self.conv_transpose1d_V(
            source_w_masked
        )  # (batch_size * n_sources, 1, T_samples_est)

        if est_source.shape[-1] > original_mixture_len:
            est_source = est_source[..., :original_mixture_len]
        elif est_source.shape[-1] < original_mixture_len:
            padding_needed = original_mixture_len - est_source.shape[-1]
            est_source = func.pad(est_source, (0, padding_needed))

        return est_source  # (batch_size*n_sources, 1, T_samples)


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization as used in Conv-TasNet.

    Performs normalization across all dimensions except the batch dimension,
    computing global statistics across time and channel dimensions.

    Args:
        channel_size (int): Number of channels to normalize
    """

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        x = (x - mean) / (var + 1e-8).sqrt() * self.gamma + self.beta
        return x


class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization for causal speech separation.

    This normalization technique is designed for causal (online) processing where
    future frames are not available. It computes statistics using only the current
    and past frames, making it suitable for real-time applications.

    The normalization is computed cumulatively across time:
    - Mean: Cumulative average of all past and current frames
    - Variance: Cumulative variance of all past and current frames

    This ensures that the normalization at time t only depends on frames 0 to t,
    maintaining causality for online processing scenarios.

    Args:
        channel_size (int): Number of channels to normalize

    Shape:
        - Input: (batch_size, channel_size, time_steps)
        - Output: (batch_size, channel_size, time_steps)
    """

    def __init__(self, channel_size):
        super(CumulativeLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x):
        """
        Forward pass with cumulative normalization.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Normalized tensor with same shape as input
        """
        # Cumulative mean: average of all frames up to current time step
        mean = torch.cumsum(x.sum(dim=1, keepdim=True), dim=2) / (
            x.shape[1] * (torch.arange(x.shape[2], device=x.device).view(1, 1, -1) + 1)
        )
        # Cumulative variance: variance of all frames up to current time step
        var = torch.cumsum((x - mean).pow(2).sum(dim=1, keepdim=True), dim=2) / (
            x.shape[1] * (torch.arange(x.shape[2], device=x.device).view(1, 1, -1) + 1)
        )
        # Apply normalization with learnable scale and shift parameters
        x = (x - mean) / (var + 1e-8).sqrt() * self.gamma + self.beta
        return x


class TemporalConvNetBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block for Conv-TasNet.

    This is the core building block of the separator network in Conv-TasNet.
    Each TCN block consists of:
    1. 1x1 convolution for channel expansion
    2. Depthwise separable convolution for temporal modeling
    3. 1x1 convolution for channel compression (bottleneck)
    4. Residual connection
    5. Optional skip connection for multi-scale feature aggregation

    The block supports both causal and non-causal convolutions, and different
    normalization types (Global LayerNorm, Cumulative LayerNorm, BatchNorm).

    Args:
        b (int): Number of bottleneck channels (input/output channels)
        h (int): Number of hidden channels in the block
        p (int): Kernel size for depthwise convolution
        sc (int): Number of skip connection channels (0 to disable)
        norm_type (str): Type of normalization ('gLN', 'cLN', or 'BN')
        causal (bool): Whether to use causal convolution (for online processing)

    Shape:
        - Input: (batch_size, b, time_frames)
        - Output: (batch_size, b, time_frames), Optional skip: (batch_size, sc, time_frames)
    """

    def __init__(
        self,
        b=config.B_TCN_CHANNELS,
        h=config.H_TCN_CHANNELS,
        p=config.P_TCN_KERNEL_SIZE,
        sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(TemporalConvNetBlock, self).__init__()

        # 1x1 convolution for channel expansion
        self.conv1x1_1 = nn.Conv1d(b, h, 1)
        self.prelu1 = nn.PReLU()

        # Choose normalization type
        if norm_type == "gLN":
            self.norm1 = GlobalLayerNorm(h)
        elif norm_type == "cLN":
            self.norm1 = CumulativeLayerNorm(h)
        else:  # BN
            self.norm1 = nn.BatchNorm1d(h)

        # Depthwise separable convolution (padding handled in forward pass)
        self.depthwise_conv = nn.Conv1d(h, h, kernel_size=p, padding=0, groups=h)
        self.prelu2 = nn.PReLU()

        # Second normalization layer
        if norm_type == "gLN":
            self.norm2 = GlobalLayerNorm(h)
        elif norm_type == "cLN":
            self.norm2 = CumulativeLayerNorm(h)
        else:  # BN
            self.norm2 = nn.BatchNorm1d(h)

        # 1x1 convolution for channel compression (bottleneck)
        self.conv1x1_2 = nn.Conv1d(h, b, 1)
        self.causal = causal

        # Optional skip connection
        if sc > 0:
            self.skip_conv1x1 = nn.Conv1d(h, sc, 1)
            self.has_skip = True
        else:
            self.has_skip = False

    def forward(self, x):
        """
        Forward pass through TCN block.

        Implements the complete TCN block processing:
        1. Channel expansion with 1x1 conv + PReLU + normalization
        2. Depthwise separable convolution with causal/non-causal padding
        3. Channel compression with 1x1 conv
        4. Residual connection
        5. Optional skip connection output

        Args:
            x: Input tensor of shape (batch, b, T_frames)

        Returns:
            tuple: (output, skip_output)
                - output: Processed tensor with residual connection (batch, b, T_frames)
                - skip_output: Skip connection tensor (batch, sc, T_frames) or None
        """
        # Store input for residual connection
        residual = x

        # 1x1 convolution for channel expansion
        out = self.conv1x1_1(x)
        out = self.prelu1(out)
        out = self.norm1(out)

        # Calculate padding for depthwise convolution to maintain temporal dimension
        kernel_size_p = self.depthwise_conv.kernel_size[0]
        if self.causal:
            # Causal padding: Only pad on the left (past) to prevent future information leakage
            pad_left = kernel_size_p - 1
            pad_right = 0
        else:
            # Non-causal "same" padding: Distribute padding symmetrically
            pad_left = (kernel_size_p - 1) // 2
            pad_right = (kernel_size_p - 1) - pad_left

        # Apply padding and depthwise convolution
        out = func.pad(out, (pad_left, pad_right))
        out = self.depthwise_conv(out)
        out = self.prelu2(out)
        out = self.norm2(out)

        # Generate skip connection output if enabled
        if self.has_skip:
            skip_out = self.skip_conv1x1(out)
        else:
            skip_out = None

        # 1x1 convolution for channel compression and residual connection
        out = self.conv1x1_2(out)
        out = out + residual  # Residual connection

        return out, skip_out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) - Separator module for Conv-TasNet.

    This is the core separator network that processes the encoded mixture representation
    to generate source separation masks. The network consists of:
    1. Layer normalization and bottleneck convolution
    2. Multiple repeats of TCN blocks with increasing dilation
    3. Skip connections for multi-scale feature aggregation
    4. Final mask generation for each source

    The TCN uses dilated convolutions to capture long-range temporal dependencies
    efficiently. Each repeat contains x blocks, and dilation increases exponentially
    within each repeat (1, 2, 4, ..., 2^(x-1)).

    Args:
        n (int): Number of encoder filters (input feature dimension)
        b (int): Number of bottleneck channels in TCN blocks
        h (int): Number of hidden channels in TCN blocks
        p (int): Kernel size for depthwise convolution
        x (int): Number of TCN blocks per repeat
        r (int): Number of repeats (each with x blocks)
        c (int): Number of sources to separate
        sc (int): Number of skip connection channels
        norm_type (str): Type of normalization ('gLN', 'cLN', or 'BN')
        causal (bool): Whether to use causal convolution

    Shape:
        - Input: (batch_size, n, time_frames) - Encoded mixture representation
        - Output: (batch_size, c, n, time_frames) - Separation masks for each source
    """

    def __init__(
        self,
        n=config.N_ENCODER_FILTERS,
        b=config.B_TCN_CHANNELS,
        h=config.H_TCN_CHANNELS,
        p=config.P_TCN_KERNEL_SIZE,
        x=config.X_TCN_BLOCKS,
        r=config.R_TCN_REPEATS,
        c=config.N_SOURCES,
        sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(TemporalConvNet, self).__init__()
        self.N = n
        self.C = c
        self.Sc = sc

        self.layer_norm = nn.LayerNorm(n)
        self.bottleneck_conv1x1 = nn.Conv1d(n, b, 1)

        self.repeats: nn.ModuleList[nn.ModuleList[TemporalConvNetBlock]] = (
            nn.ModuleList()
        )
        for _ in range(r):
            blocks: nn.ModuleList[TemporalConvNetBlock] = nn.ModuleList()
            for _ in range(x):
                blocks.append(TemporalConvNetBlock(b, h, p, sc, norm_type, causal))
            self.repeats.append(blocks)

        self.prelu_out = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(sc if sc > 0 else b, c * n, 1)

    def forward(self, mixture_w):
        # mixture_w shape: (batch, N_encoder_filters, T_frames) - output of Encoder

        # 1. Initial LayerNorm and Bottleneck Convolution
        # These layers (self.layer_norm, self.bottleneck_conv1x1) must be defined in __init__
        # Transpose to (batch, T_frames, N_encoder_filters) for LayerNorm
        processed_w = mixture_w.transpose(1, 2)
        processed_w = self.layer_norm(processed_w)
        # Transpose back to (batch, N_encoder_filters, T_frames)
        processed_w = processed_w.transpose(1, 2)
        processed_w = self.bottleneck_conv1x1(
            processed_w
        )  # Shape: (batch, B_tcn_channels, T_frames)

        skip_outputs_list = []  # To store skip_outputs if self.Sc > 0
        current_block_input = processed_w

        # 2. TCN Repeats and Blocks
        # Iterate directly over the ModuleLists.
        # The explicit type hint for 'repeat_module_list_instance' (nn.ModuleList[TemporalConvNetBlock]) is removed
        # as we are now using cast for better type inference by the analyzer.

        for repeat_module_list_instance in self.repeats:
            # repeat_module_list_instance is an nn.ModuleList of TemporalConvNetBlock.
            iterable_tcn_blocks_in_repeat = cast(
                Iterable[TemporalConvNetBlock], repeat_module_list_instance
            )

            tcn_block_module: (
                TemporalConvNetBlock  # Type hint for the inner loop variable
            )
            for (
                tcn_block_module
            ) in iterable_tcn_blocks_in_repeat:  # Iterate over casted version
                # tcn_block_module is a TemporalConvNetBlock

                residual_output, skip_output = tcn_block_module(current_block_input)

                if self.Sc > 0 and skip_output is not None:
                    skip_outputs_list.append(skip_output)

                current_block_input = (
                    residual_output  # Output of this block is input to the next
                )

        # Determine input for mask generation based on skip connections
        if self.Sc > 0 and skip_outputs_list:
            # Sum all collected skip connections
            # Ensure all skip_outputs in the list have the same shape (Batch, SC, T_frames)
            mask_input = torch.sum(torch.stack(skip_outputs_list, dim=0), dim=0)
        else:  # No skip connections (self.Sc == 0) or no skip outputs were generated
            mask_input = current_block_input  # Use the output of the last TCN block

        # 3. Mask Generation
        # These layers (self.output_prelu, self.mask_conv1x1) must be defined in __init__
        # self.mask_conv1x1 should map from (B_tcn_channels or Sc_tcn_channels) to (N_sources * N_encoder_filters) channels
        mask_output = self.prelu_out(mask_input)  # Corrected from self.output_prelu
        masks = self.mask_conv1x1(
            mask_output
        )  # Shape: (batch, N_sources * N_encoder_filters, T_frames)

        # Reshape masks to (batch, N_sources, N_encoder_filters, T_frames)
        batch_size, _, t_frames = masks.shape
        # Ensure self.C and self.N are available from __init__
        masks = masks.view(
            batch_size, self.C, self.N, t_frames
        )  # Corrected from self.n_sources and self.n_encoder_filters

        return masks


class ConvTasNet(nn.Module):
    """
    Conv-TasNet: Time-domain Audio Separation Network using Convolutional Encoder-Decoder.

    Conv-TasNet is a state-of-the-art neural network architecture for single-channel
    speech separation in the time domain. The network consists of three main components:

    1. **Encoder**: Converts time-domain waveform to high-dimensional representation
       using 1D convolution with basis functions learned during training.

    2. **Separator**: Temporal Convolutional Network (TCN) that processes the encoded
       representation to generate separation masks for each source. Uses dilated
       convolutions, residual connections, and skip connections for efficient
       long-range temporal modeling.

    3. **Decoder**: Converts the masked encoded representations back to time-domain
       waveforms using learned basis functions (transpose of encoder).

    Key Features:
    - End-to-end trainable in time domain (no STFT required)
    - Causal and non-causal processing modes
    - Efficient temporal modeling with dilated convolutions
    - Multi-scale feature aggregation via skip connections
    - Flexible normalization options (Global LayerNorm, Cumulative LayerNorm, BatchNorm)

    Args:
        n (int): Number of encoder filters (basis functions)
        l (int): Length of encoder filters (kernel size)
        b (int): Number of bottleneck channels in TCN
        h (int): Number of hidden channels in TCN blocks
        p (int): Kernel size for depthwise convolution in TCN
        x (int): Number of TCN blocks per repeat
        r (int): Number of repeats in TCN
        c (int): Number of sources to separate
        sc (int): Number of skip connection channels
        norm_type (str): Normalization type ('gLN', 'cLN', or 'BN')
        causal (bool): Whether to use causal convolution for online processing

    Shape:
        - Input: (batch_size, 1, samples) - Single-channel mixture waveform
        - Output: (batch_size, c, samples) - Separated source waveforms

    Example:
        >>> model = ConvTasNet(n=256, l=20, c=2)  # Separate 2 sources
        >>> mixture = torch.randn(4, 1, 32000)    # 4 mixtures, 2 seconds at 16kHz
        >>> sources = model(mixture)              # Shape: (4, 2, 32000)
    """

    def __init__(
        self,
        n=config.N_ENCODER_FILTERS,
        l=config.L_CONV_KERNEL_SIZE,
        b=config.B_TCN_CHANNELS,
        h=config.H_TCN_CHANNELS,
        p=config.P_TCN_KERNEL_SIZE,
        x=config.X_TCN_BLOCKS,
        r=config.R_TCN_REPEATS,
        c=config.N_SOURCES,
        sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(ConvTasNet, self).__init__()
        self.C = c

        # Initialize the three main components
        self.encoder = Encoder(l, n)
        self.separator = TemporalConvNet(n, b, h, p, x, r, c, sc, norm_type, causal)
        self.decoder = Decoder(n, l)

    def forward(self, mixture):
        """
        Forward pass through the complete Conv-TasNet architecture.

        Implements the full source separation pipeline:
        1. Encode the mixture waveform to high-dimensional representation
        2. Generate separation masks using the TCN separator
        3. Apply masks to the encoded mixture for each source
        4. Decode each masked representation back to time-domain waveforms

        The separation process follows the masking-based approach where each
        source is estimated by element-wise multiplication of the encoded mixture
        with its corresponding separation mask.

        Args:
            mixture: Input mixture waveform (batch_size, 1, samples)
                    Single-channel audio containing multiple overlapping sources

        Returns:
            estimated_sources: Separated source waveforms (batch_size, c, samples)
                              Each source is reconstructed to have the same length
                              as the original mixture input

        Processing Steps:
            1. mixture → encoder → mixture_w (encoded representation)
            2. mixture_w → separator → masks (c separation masks)
            3. mixture_w * mask_i → masked_w_i (for each source i)
            4. masked_w_i → decoder → estimated_source_i (for each source i)
            5. Stack all estimated sources into final output tensor
        """
        # Store original length for proper reconstruction
        original_mixture_len = mixture.shape[-1]

        # Step 1: Encode mixture to high-dimensional representation
        mixture_w = self.encoder(mixture)

        # Step 2: Generate separation masks for all sources
        masks = self.separator(mixture_w)

        # Step 3: Apply masks to encoded mixture for each source
        source_w_masked_list = []
        for i in range(self.C):
            mask_i = masks[:, i, :, :]  # Extract mask for source i
            masked_w = mixture_w * mask_i  # Element-wise multiplication
            source_w_masked_list.append(masked_w)

        # Step 4: Decode each masked representation back to time domain
        estimated_sources_list = []
        for masked_w_single_source in source_w_masked_list:
            est_source_single = self.decoder(
                masked_w_single_source, original_mixture_len
            )
            estimated_sources_list.append(est_source_single)

        # Step 5: Stack all sources into final output tensor
        estimated_sources = torch.stack(estimated_sources_list, dim=1)
        estimated_sources = estimated_sources.squeeze(2)

        return estimated_sources
