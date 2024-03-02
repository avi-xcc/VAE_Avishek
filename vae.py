from collections import OrderedDict

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'

class MLP(nn.Module):
    def __init__(self, hidden_sizes, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_sizes) - 1):
            in_dim = hidden_sizes[i]
            out_dim = hidden_sizes[i + 1]
            q.append((f"Linear_{i}", nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_sizes) - 2) or ((i == len(hidden_sizes) - 2) and last_activation):
                q.append((f"BatchNorm_{i}", nn.BatchNorm1d(out_dim)))
                q.append((f"ReLU_{i}", nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w - 8) // 2 - 4) // 2
        hh = ((h - 8) // 2 - 4) // 2

        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding=0), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 32, 5, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Flatten(start_dim=1),
                                    MLP([ww * hh * 64, 256, 128]))
        self.calc_mean = MLP([128 + ncond, 64, nhid], last_activation=False)
        self.calc_logvar = MLP([128 + ncond, 64, nhid], last_activation=False)

    def forward(self, x, y=None):
        x = self.encode(x)
        if y is None:
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))


class Decoder(nn.Module):
    def __init__(self, shape, nhid=16, ncond=0):
        super(Decoder, self).__init__()
        self.dim = nhid
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid + ncond, 64, 128, 256, c * w * h], last_activation=False), nn.Sigmoid())

    def forward(self, z, y=None):
        c, w, h = self.shape
        if y is None:
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)


class VAE(nn.Module):
    def __init__(self, shape, nhid=16):
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, batch_size=None):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        return res


class cVAE(nn.Module):
    def __init__(self, shape, nClass1, nClass2=None, nClass3=None, nhid=16, ncond=16):
        super(cVAE, self).__init__()

        self.label1_encoder = nn.Embedding(nClass1, ncond)
        if nClass2:
            self.label2_encoder = nn.Embedding(nClass2, ncond)
            self.label2 = True
        else:
            self.label2 = False

        if nClass3:
            self.label3_encoder = nn.Embedding(nClass3, ncond)
            self.label3 = True
        else:
            self.label3 = False

        total_ncond = ncond
        if nClass2:
            total_ncond += ncond
        if nClass3:
            total_ncond += ncond

        self.dim = nhid  # + total_ncond

        self.encoder = Encoder(shape, nhid=nhid, ncond=total_ncond)
        self.decoder = Decoder(shape, nhid=nhid, ncond=total_ncond)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y1, y2=None, y3=None):
        y = self.label1_encoder(y1)
        if y2 is not None and self.label2:
            y = torch.cat((y, self.label2_encoder(y2)), dim=1)
        if y3 is not None and self.label3:
            y = torch.cat((y, self.label3_encoder(y3)), dim=1)

        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, age, gender=None, race=None):
        age = torch.tensor(age).to(device)
        if gender is not None:
            gender = torch.tensor(gender).to(device)
        if race is not None:
            race = torch.tensor(race).to(device)

        z = torch.randn((len(age), self.dim)).to(device)
        y = self.label1_encoder(age)
        if gender is not None:
            y = torch.cat((y, self.label2_encoder(gender)), dim=1)
        if race is not None:
            y = torch.cat((y, self.label3_encoder(race)), dim=1)

        res = self.decoder(z, y)
        return res


def vae_loss(X, x_hat, mean, logvar):
    # reconstruction_loss = nn.BCELoss(reduction='sum')
    # reconstruction_loss = nn.MSELoss(reduction='sum')
    reconstruction_loss = nn.HuberLoss(reduction='sum')
    KL_div = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    return reconstruction_loss(x_hat, X) + KL_div, reconstruction_loss(x_hat, X), KL_div
