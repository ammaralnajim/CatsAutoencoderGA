import wandb
from typing import *
from tqdm.auto import tqdm
from random import randint, random, choice, choices

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import ATTEMPTS_INDIVIDUAL
from src.config import ACTIVATION_FUNCTIONS, DEVICE, BEST_MODEL_PATH, CONV_FEATURES, KERNEL_SIZES, USE_CONV, \
    LINEAR_FEATURES, LATENT_SIZE, WANDB_LOGIN
from src.autoencoder import *
from src.utils import visualize_collection


class Gene:
    def __init__(self, layer_type: str, features: int = None, kernel_size: int = None):
        """
        Class for easier work with chosen representation

        :param layer_type: 'f' for linear (fully connected), 'c' for convolutional, any from ACTIVATION_FUNCTIONS for activation funtion
        :param features: number of features outputted by layer
        :param kernel_size: used and specified for convolutions only
        """
        self.layer_type = layer_type
        self.features = features
        self.kernel_size = kernel_size

    def copy(self):
        return Gene(self.layer_type, self.features, self.kernel_size)

    def __str__(self):
        parts = [self.layer_type]
        if self.features is not None:
            parts.append(str(self.features))
        if self.kernel_size is not None:
            parts.append(str(self.kernel_size))

        return '_'.join(parts)

    def __repr__(self):
        return str(self)


class Individual:
    def __init__(self, parent, chromosomes: List[Gene] = None):
        """
        Representation of the individual in population

        :param parent: Genetic Algorithm acts as an environment for individual to act on
        :param chromosomes: List of genes representing this individual if None then random individual is created
        """
        self.parent = parent
        self.fitness = None
        if chromosomes is None:
            self.chromosomes = [Gene(choice(ACTIVATION_FUNCTIONS))]
            if USE_CONV:
                n_conv = randint(0, self.parent.max_convs)
                features = sorted(choices(CONV_FEATURES, k=n_conv))
                kernel_sizes = sorted(choices(KERNEL_SIZES, k=n_conv), reverse=True)
                for fout, ksize in zip(features, kernel_sizes):
                    self.chromosomes.append(Gene('c', fout, ksize))

            n_linear = randint(1, 5)
            features = sorted(choices(LINEAR_FEATURES, k=n_linear), reverse=True) + [LATENT_SIZE]
            for fout in features:
                self.chromosomes.append(Gene('f', fout))
        else:
            self.chromosomes = chromosomes.copy()

    def get_fitness(self, epochs):
        if self.fitness is None:
            scores = []
            for _ in range(ATTEMPTS_INDIVIDUAL):
                _, score = self.fit_autoencoder(epochs)
                scores.append(score)
            self.fitness = max(scores)

        return self.fitness

    def mutation(self, x: List[Gene]):
        """
        List of possible mutations
        1. Attempt to change activation function
        2. Delete gene
        3. Change number of features
        4. Add gene in between
        """

        # 1
        mutated_x = []
        if random() < 0.05:
            mutated_x.append(Gene(choice(ACTIVATION_FUNCTIONS)))
        else:
            mutated_x.append(x[0])

        # 2
        for i in range(1, len(x)):
            if random() > 0.01:  # so with probability 0.01 we won't add a layer
                mutated_x.append(x[i].copy())
        if len(mutated_x) == 1:  # accidentally deleted all layers
            return x

        # 3
        for i in range(1, len(mutated_x)):
            if random() < 0.05:
                if random() < 0.5:
                    mutated_x[i].features -= mutated_x[i].features // 8
                else:
                    mutated_x[i].features += mutated_x[i].features // 8

        # 4
        mutated_x2 = [mutated_x[0], mutated_x[1]]
        for i in range(2, len(mutated_x)):
            if mutated_x[i - 1].layer_type == mutated_x[i].layer_type and random() < 0.05:
                layer_type = mutated_x[i].layer_type
                features = (mutated_x[i - 1].features + mutated_x[i].features) // 2
                kernel_size = choice((mutated_x[i - 1].kernel_size, mutated_x[i].kernel_size))
                mutated_x2.append(Gene(layer_type, features, kernel_size))

            mutated_x2.append(mutated_x[i])

        return self.maintain_restrictions(mutated_x2)

    def maintain_restrictions(self, x: List[Gene]):
        """
        The list of restrictions:
        1. Convolutions strictly before fully connected layers
        2. Gradually decreasing number of features
        3. Number of convs does not exceed self.max_convs
        4. Last layer is linear with latent_size features
        """

        # fix restriction 1 via removing any [f,c,f] and [c,f,c] sequences
        rule_1_x = [x[0]]
        for i in range(1, len(x)):
            if rule_1_x[-1].layer_type == 'f' and x[i].layer_type == 'c':
                continue
            rule_1_x.append(x[i])

        # fix restriction 2 via removing increasing sequences
        rule_2_x = [rule_1_x[0], rule_1_x[1]]

        max_features = rule_1_x[1].features
        min_features = rule_1_x[1].features
        for i in range(2, len(rule_1_x)):
            current_features = rule_1_x[i].features
            if rule_1_x[i - 1].layer_type == 'c' and rule_1_x[i].layer_type == 'f':
                min_features = current_features
                rule_2_x.append(rule_1_x[i])
                continue

            if rule_1_x[i].layer_type == 'f' and min_features >= current_features:
                min_features = current_features
                rule_2_x.append(rule_1_x[i])
            elif rule_1_x[i].layer_type == 'c' and max_features <= current_features:
                max_features = current_features
                rule_2_x.append(rule_1_x[i])

        rule_3_x = [rule_2_x[0]]
        conv_limit = self.parent.max_convs
        for i in range(1, len(rule_2_x)):
            if rule_2_x[i].layer_type == 'f':
                rule_3_x.append(rule_2_x[i])
            elif rule_2_x[i].layer_type == 'c' and conv_limit > 0:
                rule_3_x.append(rule_2_x[i])
                conv_limit -= 1

        if rule_3_x[-1].layer_type != 'f':
            rule_3_x.append(Gene('f', LATENT_SIZE))
        elif rule_3_x[-1].features != LATENT_SIZE:
            rule_3_x[-1] = Gene('f', LATENT_SIZE)

        return rule_3_x

    def crossover(self, other):
        p1 = randint(1, len(self.chromosomes) - 1)
        p2 = randint(1, len(other.chromosomes) - 1)

        child1 = self.maintain_restrictions(self.chromosomes[:p1] + other.chromosomes[p2:])
        child2 = self.maintain_restrictions(other.chromosomes[:p2] + self.chromosomes[p1:])

        return Individual(self.parent, self.mutation(child1)), Individual(self.parent, self.mutation(child2))

    def create_model(self) -> nn.Module:
        """
        implements algorithm for creating trainable model from specified genome
        :return: nn.Module autoencoder for further training
        """

        encoder = []
        decoder = []
        activation = self.chromosomes[0].layer_type.lower()

        img_size = self.parent.img_shape[-1]
        img_channels = self.parent.img_shape[0]
        for i in range(1, len(self.chromosomes)):
            prev_gene = self.chromosomes[i - 1]
            gene = self.chromosomes[i]
            if gene.layer_type == 'c':
                if i == 1:
                    encoder.append(ConvLayer(img_channels,
                                             gene.features,
                                             gene.kernel_size,
                                             activation))
                    decoder.append(ConvTransposeLayer(gene.features,
                                                      img_channels,
                                                      gene.kernel_size,
                                                      activation))
                else:
                    encoder.append(ConvLayer(prev_gene.features,
                                             gene.features,
                                             gene.kernel_size,
                                             activation))
                    decoder.append(ConvTransposeLayer(gene.features,
                                                      prev_gene.features,
                                                      gene.kernel_size,
                                                      activation))
                img_size //= 2
                img_channels = gene.features

            elif gene.layer_type == 'f':
                if i == 1 or prev_gene.layer_type == 'c':
                    encoder.append(LinearLayer(img_channels * img_size * img_size,
                                               gene.features,
                                               activation if i != len(self.chromosomes) - 1 else None,
                                               flatten=True))
                    decoder.append(LinearLayer(gene.features,
                                               img_channels * img_size * img_size,
                                               activation,
                                               unflatten=(img_channels, img_size, img_size)))

                else:
                    encoder.append(LinearLayer(prev_gene.features,
                                               gene.features,
                                               activation if i != len(self.chromosomes) - 1 else None))
                    decoder.append(LinearLayer(gene.features,
                                               prev_gene.features,
                                               activation))
        decoder = decoder[::-1]
        img_channels = self.parent.img_shape[0]
        decoder.append(nn.Sequential(nn.Conv2d(img_channels, img_channels, 3, padding=1), nn.Sigmoid()))

        return Autoencoder(encoder, decoder)

    def fit_epoch(self, model, criterion, optimizer, scheduler):
        """
        traning part of training loop

        :param model: model
        :param criterion: loss function
        :param optimizer:
        :param scheduler:
        :return: averaged training loss
        """
        model.train()
        running_loss = 0.0

        for inputs in self.parent.train_loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()

            latent_vector, reconstruction = model(inputs)

            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(self.parent.train_loader)
        return train_loss

    def eval_epoch(self, model, criterion):
        """
        validation part of training loop

        :param model: model
        :param criterion: loss function
        :return: averaged loss on validation set
        """
        model.eval()
        running_loss = 0.0

        for inputs in self.parent.valid_loader:
            inputs = inputs.to(DEVICE)
            with torch.set_grad_enabled(False):
                latent_vector, reconstruction = model(inputs)
                loss = criterion(reconstruction, inputs)

            running_loss += loss.item()

        valid_loss = running_loss / len(self.parent.valid_loader)
        return valid_loss

    def train(self, model, criterion, optimizer, scheduler, epochs=50, patience=10, verbose=True):
        """
        training loop

        :param model: model
        :param criterion: loss function
        :param optimizer:
        :param scheduler:
        :param epochs: number of epochs
        :param patience:
        :param verbose: whether to output messages and progress bars
        :return: best model and best validation score (loss)
        """
        best_score = np.inf

        if not verbose:
            pbar = range(epochs)
        else:
            pbar = tqdm(list(range(epochs)), desc=str(self.chromosomes))

        for epoch in pbar:
            train_loss = self.fit_epoch(model, criterion, optimizer, scheduler)

            valid_loss = self.eval_epoch(model, criterion)

            if valid_loss < best_score:
                best_score = valid_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                patience -= 1

            if patience == 0:
                if verbose:
                    print("Early stopping!")
                break
            pbar.set_postfix_str(f"Train loss: {train_loss}, Validation loss: {valid_loss}")

        model.load_state_dict(torch.load(BEST_MODEL_PATH))

        return model, best_score

    def get_latent_dist(self, model):
        """
        :param model:
        :return: latent distribution of encoder output for given model
        """
        model.eval()

        latent_vectors = None
        for inputs in self.parent.train_loader:
            inputs = inputs.to(DEVICE)
            with torch.set_grad_enabled(False):
                latent_vector = model.encoder(inputs).cpu().detach()
                if latent_vectors is None:
                    latent_vectors = latent_vector
                else:
                    latent_vectors = torch.concat((latent_vectors, latent_vector), dim=0)

        return torch.mean(latent_vectors, dim=0), torch.std(latent_vectors, dim=0)

    def fit_autoencoder(self, epochs):
        """
        trains autoencoder when genetic algorithm in the train phase
        :param epochs:
        :return: best model and best score
        """
        model = self.create_model()
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-5)
        model.to(DEVICE)
        model, best_score = self.train(model, criterion, optimizer, scheduler, epochs, -1)
        return model.to('cpu'), best_score

    def fit_long(self, epochs):
        """
        trains best individual after genetic algorithm is finished
        :param epochs:
        :return: best model and best score
        """
        model = self.create_model()
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
        model.to(DEVICE)
        model, best_score = self.train(model, criterion, optimizer, scheduler, epochs, epochs // 2)
        return model.to('cpu'), best_score

    def visualize(self, epochs, n=16):
        """
        Visualizes generation capabilities of individual

        :param epochs:
        :param n: number of output images for correct work make it square of natural number
        :return:
        """
        model, best_score = self.fit_long(epochs)
        means, stds = self.get_latent_dist(model.to(DEVICE))
        model.eval()
        latent_batch = torch.concat([torch.normal(means, stds).unsqueeze(0) for _ in range(n)], dim=0).to(DEVICE)
        recs = model.decoder(latent_batch)
        model.to('cpu')

        visualize_collection(recs, n)

    def __str__(self):
        return str(self.chromosomes)

    def __repr__(self):
        return str(self)


class GeneticAlgorithm:
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 batch_size=8):
        """
        Whole class dedicated to encapsulate logic of training genetic algorithm for given task
        :param train_dataset:
        :param valid_dataset:
        :param batch_size:
        """
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
        self.img_shape = train_dataset[0].shape

        self.max_convs = 0
        img_size = self.img_shape[-1]
        while img_size > 2:
            self.max_convs += 1
            img_size //= 2

    def train_ga(self, n_pop: int, p_new: float, epochs, epochs_per_model):
        """
        Main loop of genetic algorithm
        :param n_pop: number of initial population
        :param p_new: share of 'children' created at each step that will replace old ones
        :param epochs: number of iterations of genetic algorithm
        :param epochs_per_model: number of iterations per individual
        :return:
        """
        if WANDB_LOGIN:
            wandb.init(
                project="ga-autoencoders-cats",
                config={
                    "n_pop": n_pop,
                    "p_new": p_new,
                    "ga_epochs": epochs,
                    "model_epochs": epochs_per_model,
                    "batch_size": self.batch_size,
                    "latent_size": LATENT_SIZE
                }
            )

        population = [Individual(self) for _ in range(n_pop)]

        n_children = int(n_pop * p_new)
        n_children -= n_children % 2

        population.sort(key=lambda x: x.get_fitness(epochs_per_model), reverse=True)

        best_fit = population[-1]
        best_fitness = population[-1].get_fitness(epochs_per_model)

        if WANDB_LOGIN:
            wandb.log({"best_score": best_fitness})

        pbar = tqdm(list(range(epochs)), desc='Genetic algorithm')
        for _ in pbar:
            best_pops = population[-n_children // 2:]
            candidate_pops = population[-n_children:]
            children = []
            for ind in best_pops:
                child1, child2 = ind.crossover(choice(candidate_pops))
                children.append(child1)
                children.append(child2)

            population[:n_children] = children

            population.sort(key=lambda x: x.get_fitness(epochs_per_model), reverse=True)

            if population[-1].get_fitness(epochs_per_model) < best_fitness:
                best_fitness = population[-1].get_fitness(epochs_per_model)
                best_fit = population[-1]

            pbar.set_postfix_str(f'Best score: {best_fitness}, Best_individual: {best_fit}')
            if WANDB_LOGIN:
                wandb.log({"best_score": best_fitness})
        if WANDB_LOGIN:
            wandb.finish()
        return best_fit, best_fitness
