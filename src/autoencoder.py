from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, fin, fout, kernel_size, activation):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(fin, fout, kernel_size, padding=padding),
            nn.BatchNorm2d(fout)
        ]

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation is None:
            pass
        else:
            raise AttributeError(
                f"Invalid value for activation parameter valid values are ['relu', 'lrelu', 'tanh', 'sigmoid', None], got {activation}")

        layers.append(nn.MaxPool2d(2))

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class ConvTransposeLayer(nn.Module):
    def __init__(self, fin, fout, kernel_size, activation):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2),
        ]

        padding = kernel_size // 2
        layers.append(nn.Conv2d(fin, fout, kernel_size, padding=padding))
        layers.append(nn.BatchNorm2d(fout))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation is None:
            pass
        else:
            raise AttributeError(
                "Invalid value for activation parameter valid values are ['relu', 'lrelu', 'tanh', 'sigmoid', None]")

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class LinearLayer(nn.Module):
    def __init__(self, fin, fout, activation, flatten=False, unflatten=None):
        super().__init__()
        self.unflatten = unflatten

        layers = []
        if flatten:
            layers.append(nn.Flatten())

        layers.append(nn.Linear(fin, fout))
        layers.append(nn.BatchNorm1d(fout))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation is None:
            layers.pop()
        else:
            raise AttributeError(
                "Invalid value for activation parameter valid values are ['relu', 'lrelu', 'tanh', 'sigmoid', None]")

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        x = self.module(x)
        if self.unflatten is not None:
            x = x.view(-1, *self.unflatten)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)
        return latent_vector, reconstruction
