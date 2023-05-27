from torchvision import transforms as tt

WANDB_LOGIN = False

# dataset preprocessing configs
LOAD_IN_MEMORY = False
DATASET_PATH = "./cats"
DEVICE = 'cpu'
TRANSFORM = tt.Compose([
    tt.ToTensor(),
    tt.Resize(32),
])

# training configs
TRAIN_PORTION = 0.1  # either float as portion or integer as number of objects
VALIDATION_PORTION = 0.05  # either float as portion or integer as number of objects
BATCH_SIZE = 16
ADD_NOISE = True
N_POPULATION = 4
P_CHILDREN = 0.5
EPOCHS_GA = 3
ATTEMPTS_INDIVIDUAL = 1
EPOCHS_INDIVIDUAL = 1

EPOCHS_BEST = 30

# genetic algorithm details
ACTIVATION_FUNCTIONS = ['relu', 'lrelu']  # supported relu, lrelu, tanh and sigmoid
LINEAR_FEATURES = [192, 256, 384, 512, 768, 1024, 1536, 2048]  # , 4096, 8192]
USE_CONV = False
CONV_FEATURES = [3, 4, 8, 12, 16, 24, 32]
KERNEL_SIZES = [3, 5]
LATENT_SIZE = 16
BEST_MODEL_PATH = "./results/best_model.pth"  # should be in existing directory

# output configs
ROOT_N_IMAGES = 6
GENERATED_PATH = "./results/best.png"  # should be in existing directory
TRUE_PATH = "./results/true_eval.png"  # should be in existing directory
RECS_PATH = "./results/recs_eval.png"  # should be in existing directory
