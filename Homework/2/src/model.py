import torch
import torch.nn as nn
import torch.nn.functional as F

# Import config from the same directory
import config


# --- Model Components ---
class Encoder(nn.Module):
    def __init__(self, L=config.L_CONV_KERNEL_SIZE, N=config.N_ENCODER_FILTERS):
        super(Encoder, self).__init__()
        self.conv1d_U = nn.Conv1d(
            1, N, kernel_size=L, stride=L // 2, padding=0, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, mixture):  # mixture: (batch, 1, T_samples)
        mixture_w = self.relu(self.conv1d_U(mixture))  # (batch, N, T_frames)
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N=config.N_ENCODER_FILTERS, L=config.L_CONV_KERNEL_SIZE):
        super(Decoder, self).__init__()
        self.N = N
        self.L = L
        self.conv_transpose1d_V = nn.ConvTranspose1d(
            N, 1, kernel_size=L, stride=L // 2, bias=False
        )

    def forward(
        self, source_w_masked, original_mixture_len
    ):  # source_w_masked: (batch, N, T_frames)
        est_source = self.conv_transpose1d_V(
            source_w_masked
        )  # (batch_size * n_sources, 1, T_samples_est)

        if est_source.shape[-1] > original_mixture_len:
            est_source = est_source[..., :original_mixture_len]
        elif est_source.shape[-1] < original_mixture_len:
            padding_needed = original_mixture_len - est_source.shape[-1]
            est_source = F.pad(est_source, (0, padding_needed))

        return est_source  # (batch_size*n_sources, 1, T_samples)


class GlobalLayerNorm(nn.Module):
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
    def __init__(self, channel_size):
        super(CumulativeLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x):
        mean = torch.cumsum(x.sum(dim=1, keepdim=True), dim=2) / (
            x.shape[1] * (torch.arange(x.shape[2], device=x.device).view(1, 1, -1) + 1)
        )
        var = torch.cumsum((x - mean).pow(2).sum(dim=1, keepdim=True), dim=2) / (
            x.shape[1] * (torch.arange(x.shape[2], device=x.device).view(1, 1, -1) + 1)
        )
        x = (x - mean) / (var + 1e-8).sqrt() * self.gamma + self.beta
        return x


class TemporalConvNetBlock(nn.Module):
    def __init__(
        self,
        B=config.B_TCN_CHANNELS,
        H=config.H_TCN_CHANNELS,
        P=config.P_TCN_KERNEL_SIZE,
        Sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(TemporalConvNetBlock, self).__init__()
        self.conv1x1_1 = nn.Conv1d(B, H, 1)
        self.prelu1 = nn.PReLU()
        if norm_type == "gLN":
            self.norm1 = GlobalLayerNorm(H)
        elif norm_type == "cLN":
            self.norm1 = CumulativeLayerNorm(H)
        else:  # BN
            self.norm1 = nn.BatchNorm1d(H)

        self.depthwise_conv = nn.Conv1d(
            H, H, kernel_size=P, padding=(P - 1) // 2 if not causal else P - 1, groups=H
        )
        self.prelu2 = nn.PReLU()
        if norm_type == "gLN":
            self.norm2 = GlobalLayerNorm(H)
        elif norm_type == "cLN":
            self.norm2 = CumulativeLayerNorm(H)
        else:  # BN
            self.norm2 = nn.BatchNorm1d(H)

        self.conv1x1_2 = nn.Conv1d(H, B, 1)  # Bottleneck
        self.causal = causal

        if Sc > 0:  # Skip connection
            self.skip_conv1x1 = nn.Conv1d(H, Sc, 1)
            self.has_skip = True
        else:
            self.has_skip = False

    def forward(self, x):  # x: (batch, B, T_frames)
        residual = x
        out = self.conv1x1_1(x)
        out = self.prelu1(out)
        out = self.norm1(out)

        if self.causal:
            padding_val = self.depthwise_conv.kernel_size[0] - 1
            out = F.pad(out, (padding_val, 0))
        out = self.depthwise_conv(out)
        out = self.prelu2(out)
        out = self.norm2(out)

        if self.has_skip:
            skip_out = self.skip_conv1x1(out)
        else:
            skip_out = None

        out = self.conv1x1_2(out)
        out = out + residual

        return out, skip_out


class TemporalConvNet(nn.Module):  # Separator
    def __init__(
        self,
        N=config.N_ENCODER_FILTERS,
        B=config.B_TCN_CHANNELS,
        H=config.H_TCN_CHANNELS,
        P=config.P_TCN_KERNEL_SIZE,
        X=config.X_TCN_BLOCKS,
        R=config.R_TCN_REPEATS,
        C=config.N_SOURCES,
        Sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(TemporalConvNet, self).__init__()
        self.N = N
        self.C = C
        self.Sc = Sc

        self.layer_norm = nn.LayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1)

        self.repeats = nn.ModuleList()
        for _ in range(R):
            blocks = nn.ModuleList()
            for _ in range(X):
                blocks.append(TemporalConvNetBlock(B, H, P, Sc, norm_type, causal))
            self.repeats.append(blocks)

        self.prelu_out = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(Sc if Sc > 0 else B, C * N, 1)

    def forward(self, mixture_w):
        batch_size, N, T_frames = mixture_w.shape

        out = self.layer_norm(mixture_w.transpose(1, 2)).transpose(1, 2)
        out = self.bottleneck_conv1x1(out)

        skip_connection_sum = None
        if self.Sc > 0:
            skip_connection_sum = torch.zeros(
                (batch_size, self.Sc, T_frames), device=mixture_w.device
            )

        for r_idx in range(len(self.repeats)):
            for block_idx in range(len(self.repeats[r_idx])):
                residual_out, skip_out = self.repeats[r_idx][block_idx](out)
                out = residual_out
                if skip_out is not None and skip_connection_sum is not None:
                    skip_connection_sum = skip_connection_sum + skip_out

        if skip_connection_sum is not None:
            processed_output = skip_connection_sum
        else:
            processed_output = out

        processed_output = self.prelu_out(processed_output)
        masks = self.mask_conv1x1(processed_output)
        masks = masks.view(batch_size, self.C, N, T_frames)
        masks = torch.sigmoid(masks)

        return masks


class ConvTasNet(nn.Module):
    def __init__(
        self,
        N=config.N_ENCODER_FILTERS,
        L=config.L_CONV_KERNEL_SIZE,
        B=config.B_TCN_CHANNELS,
        H=config.H_TCN_CHANNELS,
        P=config.P_TCN_KERNEL_SIZE,
        X=config.X_TCN_BLOCKS,
        R=config.R_TCN_REPEATS,
        C=config.N_SOURCES,
        Sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ):
        super(ConvTasNet, self).__init__()
        self.C = C
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, Sc, norm_type, causal)
        self.decoder = Decoder(N, L)

    def forward(self, mixture):
        original_mixture_len = mixture.shape[-1]
        mixture_w = self.encoder(mixture)

        masks = self.separator(mixture_w)

        source_w_masked_list = []
        for i in range(self.C):
            mask_i = masks[:, i, :, :]
            masked_w = mixture_w * mask_i
            source_w_masked_list.append(masked_w)

        estimated_sources_list = []
        for masked_w_single_source in source_w_masked_list:
            est_source_single = self.decoder(
                masked_w_single_source, original_mixture_len
            )
            estimated_sources_list.append(est_source_single)

        estimated_sources = torch.stack(estimated_sources_list, dim=1)
        estimated_sources = estimated_sources.squeeze(2)

        return estimated_sources
