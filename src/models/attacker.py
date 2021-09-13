import torch
from torch import nn, Tensor
import torch.nn.functional as F





class Flatten(torch.nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)

class Attacker(nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Flatten(),
            torch.nn.Linear(9216, 500),)

        self.linear1 = torch.nn.Linear(500, 9216)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=8,
                kernel_size=3,
                padding=0,
                stride=1,
                output_padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=0,
                stride=1,
                output_padding=0,
            ),
        )
        

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x, classifer_flag=True):
        
        if classifer_flag:
            x = self.encoder(x)
            return self.classifer(x)
        else:
            with torch.no_grad():
                x = self.encoder(x)
            x = self.linear1(x)
            x = x.view(-1, 64, 12, 12)
            return self.decoder(x)

    def inference(self, x):
        x = self.linear1(x)
        x = x.view(-1, 64, 12, 12)
        return self.decoder(x)