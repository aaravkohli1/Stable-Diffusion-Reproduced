import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
        self.d_heads = embd_dim // n_heads  # Dimension of each head 32 to 128

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        batch_size, seq_len, d_emed = x.shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        # Compute Q, K, V from input x
        q, k, v = self.in_proj(x).chunk(3, dim=-1)  

        # (batch_size, seq_len, d_embd) -> (batch_size, seq_len, n_heads, d_heads) for multi head attention
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)

        # (batch_size, n_heads, seq_len, d_heads)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention scores
        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        # (batch_size, n_heads, seq_len, dim / n) -> (batch_size, seq_len, n_heads, dim / n)
        output = output.transpose(1, 2)

        output = output.reshape((batch_size, seq_len, d_emed))

        output = self.out_proj(output)

        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # divides the channels into 32 groups and normalizes each group separately
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels) # 1 head should be enough as CNN does heavy lifting

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        residual = x.clone()

        x = self.groupnorm(x)
        
        n, c, h, w = x.shape

        x = x.view((n, c, h * w)).transpose(-1, -2)  # (batch_size, h*w, channels)

        # attention without mask
        x = self.attention(x)  # (batch_size, h*w, channels)

        x = x.transpose(-1, -2).view((n, c, h, w))  # (batch_size, channels, height, width)

        return x + residual
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # kernel_size = 1 so linear change in channels to fit channel dimensions
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        residual = x.clone()

        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_conv(residual)
    
class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # 3 channels -> 128 channels
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                # Adds padding to right and bottom so (h, w) -> (h+1, w+1)
                x = F.pad(x, (0, 1, 0, 1)) # (left, rigth, top, bottom) padding to handle odd dimensions
            x = module(x)
        
        # split the 8 channels into 4 for mean and 4 for log_variance
        # take log_variance instead of variance to avoid negative variance values which are not valid
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        # reparameterization trick
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std

        # normalization constant... don't ask me where this came from...
        x *= 0.18215

        return x, mean, log_variance
    
class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 4, h / 8, w / 8)
        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, h, w)
        return x
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded, mean, log_variance = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded, mean, log_variance
    
