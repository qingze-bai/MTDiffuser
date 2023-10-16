import torch
from torch import nn

class _DomainSpecificGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, num_domains: int = 1, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        super(_DomainSpecificGroupNorm, self).__init__()
        self.num_domains = num_domains
        self.gns = nn.ModuleList(
            [nn.GroupNorm(num_groups, num_channels, eps, affine, device, dtype) for _ in range(num_domains)])

    def reset_running_stats(self):
        for gn in self.gns:
            gn.reset_running_stats()

    def reset_parameters(self):
        for gn in self.gns:
            gn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        x = torch.concat([self.gns[domain_label[i]](data) for i, data in enumerate(x.split(split_size=1, dim=0))], dim=0)
        return x

class DomainSpecificGroupNorm2D(_DomainSpecificGroupNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
