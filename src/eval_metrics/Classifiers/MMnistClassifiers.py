import torch
import torch.nn as nn
import numpy as np



class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)
  

# class ClfImg(nn.Module):
#     def __init__(self):
#         super(ClfImg, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2);
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2);
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2);
#         self.relu = nn.ReLU();
#         self.d = nn.d(p=0.5, inplace=False);
#         self.linear = nn.Linear(in_features=128, out_features=10, bias=True)  # 10 is the number of classes (=digits)
#         self.sigmoid = nn.Sigmoid();

#     def forward(self, x):
#         h = self.conv1(x);
#         h = self.relu(h);
#         h = self.conv2(h);
#         h = self.relu(h);
#         h = self.conv3(h);
#         h = self.relu(h);
#         h = self.d(h);
#         h = h.view(h.size(0), -1);
#         h = self.linear(h);
#         out = self.sigmoid(h);
#         return out;

#     def get_activations(self, x):
#         h = self.conv1(x);
#         h = self.relu(h);
#         h = self.conv2(h);
#         h = self.relu(h);
#         h = self.conv3(h);
#         h = self.relu(h);
#         h = self.d(h);
#         h = h.view(h.size(0), -1);
#         return h;

class ClfImg(nn.Module):
    def __init__(self):
        super().__init__()
        s0 = self.s0 = 7
        nf = self.nf = 64
        nf_max = self.nf_max = 1024
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out
