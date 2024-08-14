import torch.nn as nn

def modulate(x, shift, scale):
    ## Create an additional dimension so that affine
    ## transformations can be applied element wise.
    ## if x: [15, 256, 1152], scale & shift: [15, 1152]
    ## unsqueeze -> [15, 1, 1152]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        ## The modulation applied here can help the network
        ## learn complex representation features
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x