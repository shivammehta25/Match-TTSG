import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from match_ttsg.models.components.decoder import (SinusoidalPosEmb,
                                                  TimestepEmbedding)
from match_ttsg.models.components.text_encoder import LayerNorm
from match_ttsg.utils.model import average_over_durations, expand_lengths

# Define available networks

class DurationPredictorNetwork(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, out=1):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, out, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class DurationPredictorNetworkWithTimeStep(nn.Module):
    """Similar architecture but with a time embedding support"""

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.time_embeddings = SinusoidalPosEmb(filter_channels)
        self.time_mlp = TimestepEmbedding(
            in_channels=filter_channels,
            time_embed_dim=filter_channels,
            act_fn="silu",
        )

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, enc_outputs, t):
        t = self.time_embeddings(t)
        t = self.time_mlp(t).unsqueeze(-1)

        x = pack([x, enc_outputs], "b * t")[0]

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = x + t
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = x + t
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


# Define available methods to compute loss

# Simple MSE deterministic


class DeterministicDurationPredictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.estimator = DurationPredictorNetwork(
            params.n_channels + (params.spk_emb_dim if params.n_spks > 1 else 0),
            params.filter_channels,
            params.kernel_size,
            params.p_dropout,
        )

    @torch.inference_mode()
    def forward(self, x, x_mask):
        return self.estimator(x, x_mask)

    def compute_loss(self, durations, enc_outputs, x_mask):
        return F.mse_loss(self.estimator(enc_outputs, x_mask), durations, reduction="sum") / torch.sum(x_mask)


# Flow Matching duration predictor


class FlowMatchingDurationPrediction(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        self.estimator = DurationPredictorNetworkWithTimeStep(
            1
            + params.n_channels
            + (
                params.spk_emb_dim if params.n_spks > 1 else 0
            ),  # 1 for the durations and n_channels for encoder outputs
            params.filter_channels,
            params.kernel_size,
            params.p_dropout,
        )
        self.sigma_min = params.sigma_min
        self.n_steps = params.n_steps

    @torch.inference_mode()
    def forward(self, enc_outputs, mask, n_timesteps=10, temperature=.667):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if n_timesteps is None:
            n_timesteps = self.n_steps

        b, _, t = enc_outputs.shape
        z = torch.randn((b, 1, t), device=enc_outputs.device, dtype=enc_outputs.dtype) * temperature
        if n_timesteps == 0:
            return z

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=enc_outputs.device)
        return self.solve_euler(z, t_span=t_span, enc_outputs=enc_outputs, mask=mask)

    def solve_euler(self, x, t_span, enc_outputs, mask):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, enc_outputs, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, enc_outputs, mask):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        enc_outputs = enc_outputs.detach()  # don't update encoder from the duration predictor
        b, _, t = enc_outputs.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=enc_outputs.device, dtype=enc_outputs.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, enc_outputs, t.squeeze()), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss



class DP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.name = params.name

        if params.name == "deterministic":
            self.dp = DeterministicDurationPredictor(
                params,
            )
        elif params.name == "flow_matching":
            self.dp = FlowMatchingDurationPrediction(
                params,
            )
        else:
            raise ValueError(f"Invalid duration predictor configuration: {params.name}")

    @torch.inference_mode()
    def forward(self, enc_outputs, mask, n_timesteps=None):
        if self.name == "flow_matching":
            return self.dp(enc_outputs, mask, n_timesteps)
        else:
            return self.dp(enc_outputs, mask)

    def compute_loss(self, durations, enc_outputs, mask):
        return self.dp.compute_loss(durations, enc_outputs, mask) 


class ProsodyPredictors(nn.Module):
    def __init__(
            self,
            params,
            n_spks,
            spk_emb_dim,
            pitch_min,
            pitch_max,
            energy_min,
            energy_max    
        ) -> None:
        super().__init__()
        self.n_spks = n_spks
        self.name = params.name
        
        if params.name == "deterministic":
            self.pitch_predictor = DurationPredictorNetwork(
                params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
                params.filter_channels,
                params.kernel_size,
                params.p_dropout
            )
            self.energy_predictor = DurationPredictorNetwork(
                params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
                params.filter_channels,
                params.kernel_size,
                params.p_dropout
            )
        else:
            raise ValueError(f"Invalid prosody predictor configuration: {params.name}")
        
        n_bins, steps = params.n_bins, params.n_bins - 1
        self.pitch_bins = nn.Parameter(torch.linspace(pitch_min.item(), pitch_max.item(), steps), requires_grad=False)
        self.embed_pitch = nn.Embedding(n_bins, params.n_channels)
        nn.init.normal_(self.embed_pitch.weight, mean=0, std=params.n_channels**-0.5)
        self.energy_bins =  nn.Parameter(torch.linspace(energy_min.item(), energy_max.item(), steps), requires_grad=False)
        self.embed_energy = nn.Embedding(n_bins, params.n_channels) 
        nn.init.normal_(self.embed_energy.weight, mean=0, std=params.n_channels**-0.5)
        

    @torch.no_grad()
    def synthesise(self, enc_outputs, durations, mask, spks=None):
        if self.n_spks > 1:
            enc_outputs = torch.cat([enc_outputs, spks.unsqueeze(-1).repeat(1, 1, enc_outputs.shape[-1])], dim=1)

        pitch_out = self.pitch_predictor(enc_outputs, mask)
        pitch_emb = self.embed_pitch(torch.bucketize(pitch_out.squeeze(1), self.pitch_bins)) * rearrange(mask, "b 1 t -> b t 1") 
        enc_outputs = enc_outputs + rearrange(pitch_emb, "b t c -> b c t")
        energy_out = self.energy_predictor(enc_outputs, mask)
        energy_emb = self.embed_energy(torch.bucketize(energy_out.squeeze(1), self.energy_bins)) * rearrange(mask, "b 1 t -> b t 1")
        enc_outputs = enc_outputs + rearrange(energy_emb, "b t c -> b c t")
        
        durations = rearrange(durations, "b 1 t -> b t") 
        pitch_out = expand_lengths(rearrange(pitch_out, "b 1 t -> b t 1" ), durations)[0]
        energy_out = expand_lengths(rearrange(energy_out, "b 1 t -> b t 1" ), durations)[0]
        
        
        return enc_outputs, {
            "pitch_pred": pitch_out, 
            "energy_pred": energy_out
        }
    
    def forward(self, enc_outputs, mask, durations, pitch, energy, spks=None):
        """

        Args:
            enc_outputs (b c t): 
            mask (b 1 t): 
            pitch (b t): 
            energy (b t): 
            spks (b s, optional): Defaults to None.

        Returns:
            enc_outputs: b c t
            losses: dict
        """
        if self.n_spks > 1:
            enc_outputs = torch.cat([enc_outputs, spks.unsqueeze(-1).repeat(1, 1, enc_outputs.shape[-1])], dim=1)

              
        pitch = average_over_durations(rearrange(pitch, "b t -> b 1 t"), durations)
        energy = average_over_durations(rearrange(energy, "b t -> b 1 t"), durations)
        
        pitch_out = self.pitch_predictor(enc_outputs, mask)
        pitch_emb = self.embed_pitch(torch.bucketize(pitch.squeeze(1), self.pitch_bins)) * rearrange(mask, "b 1 t -> b t 1")
        enc_outputs = enc_outputs + rearrange(pitch_emb, "b t c -> b c t")
        energy_out = self.energy_predictor(enc_outputs, mask)
        energy_emb = self.embed_energy(torch.bucketize(energy.squeeze(1), self.energy_bins)) * rearrange(mask, "b 1 t -> b t 1")
        enc_outputs = enc_outputs + rearrange(energy_emb, "b t c -> b c t")
  
        pitch_loss = F.mse_loss(pitch_out, pitch, reduction="sum") / torch.sum(mask)
        energy_loss = F.mse_loss(energy_out, energy, reduction="sum") / torch.sum(mask)
 
         
        losses = {
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss
        }
        return enc_outputs, losses