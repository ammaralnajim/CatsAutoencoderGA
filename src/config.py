from torchvision import transforms as tt

WANDB_LOGIN = True

# dataset preprocessing configs
LOAD_IN_MEMORY = True
DATASET_PATH = "./cats/cats"
DEVICE = 'cuda'
TRANSFORM = tt.Compose([
    tt.ToTensor(),
    tt.Resize(64),
])

# training configs
TRAIN_PORTION = 8192  # either float as portion or integer as number of objects
VALIDATION_PORTION = 2048  # either float as portion or integer as number of objects
BATCH_SIZE = 16
ADD_NOISE = True
N_POPULATION = 64
P_CHILDREN = 0.5
EPOCHS_GA = 10
ATTEMPTS_INDIVIDUAL = 2
EPOCHS_INDIVIDUAL = 2

EPOCHS_BEST = 100

# genetic algorithm details
ACTIVATION_FUNCTIONS = ['relu', 'lrelu']  # supported relu, lrelu, tanh and sigmoid
LINEAR_FEATURES = [192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192]
USE_CONV = True
CONV_FEATURES = [3, 4, 8, 12, 16, 24, 32]
KERNEL_SIZES = [3, 5]
LATENT_SIZE = 16
BEST_MODEL_PATH = "./results/best_model.pth"  # should be in existing directory

# output configs
ROOT_N_IMAGES = 6
GENERATED_PATH = "./results/best.png"  # should be in existing directory
TRUE_PATH = "./results/true_eval.png"  # should be in existing directory
RECS_PATH = "./results/recs_eval.png"  # should be in existing directory
