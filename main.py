import wandb
from torch.utils.data import random_split
from torch import Generator

from src.config import TRAIN_PORTION, VALIDATION_PORTION, BATCH_SIZE, N_POPULATION, P_CHILDREN, EPOCHS_GA, \
    EPOCHS_INDIVIDUAL, EPOCHS_BEST, LOAD_IN_MEMORY, WANDB_LOGIN
from src.genetic_algorithm import GeneticAlgorithm
from src.prepare_dataset import get_dataset


def main():
    print("Starting training...")
    if WANDB_LOGIN:
        wandb.login()
    dataset = get_dataset(LOAD_IN_MEMORY)

    if isinstance(TRAIN_PORTION, float) and isinstance(VALIDATION_PORTION, float):
        total = 1
    else:
        total = len(dataset)

    train_dataset, valid_dataset, _ = random_split(dataset,
                                                   [TRAIN_PORTION, VALIDATION_PORTION,
                                                    total - TRAIN_PORTION - VALIDATION_PORTION],
                                                   generator=Generator().manual_seed(42))

    print(f"Training count: {len(train_dataset)}")
    print(f"Validation count: {len(valid_dataset)}")

    ga = GeneticAlgorithm(train_dataset, valid_dataset, BATCH_SIZE)
    best_fit, best_score = ga.train_ga(N_POPULATION,
                                       P_CHILDREN,
                                       EPOCHS_GA,
                                       EPOCHS_INDIVIDUAL)

    best_fit.visualize(EPOCHS_BEST)


if __name__ == '__main__':
    main()
