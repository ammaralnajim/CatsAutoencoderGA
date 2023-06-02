# CatsAutoencoderGA
My own interpretation of Nature Inspired Computing final project at Innopolis University

Here you can find project [article](https://drive.google.com/file/d/13g3zIlCQYJ6vfD-62MEhq0lYbYSEyAOm/view) and original [repository](https://github.com/KGallyamov/NIC-project), which you can address for more information

## Installation
### Python
You need [Python 3.9](https://www.python.org/downloads/) or above for proper work

### Repository
Clone repository with:
```
git clone https://github.com/IGragon/CatsAutoencoderGA.git
```

### Libraries
Necessary libraries can be installed through:
```
pip install -r requirements.txt
```

## Configuration
After installation you may want to change [config file](src/config.py)

Important step is to verify that all PATH variables have existing directories

More detailed explaination of cofig varaibles located at the [footer](#configuration-details)

## Run
After you configured everything you can run training with:
```
python main.py
```

After completion there will be three images: true and recreated images from training set and generated images randomly sampled from latent space

## Configuration details

WANDB_LOGIN - whether to login to wandb for experiment tracking

### Dataset preprocessing configs
LOAD_IN_MEMORY - whether to pre-load all data to memory

DATASET_PATH - path to the images, have to contain only images

DEVICE - what device to use 'cpu' or 'cuda'

TRANSFORM - torchvision Compose object with all transforms, applied both to train and validation images

### Training configs
TRAIN_PORTION - either float as portion or integer as number of objects

VALIDATION_PORTION - either float as portion or integer as number of objects

BATCH_SIZE - batch size

ADD_NOISE - whether to add noise to input images before putting them through autoencoder in training phase

N_POPULATION - number of individuals in the initial population

P_CHILDREN - what portion of the population is replaced with children

EPOCHS_GA - number of iterations for genetic algorithm

ATTEMPTS_INDIVIDUAL - number of attempts to fit individual

EPOCHS_INDIVIDUAL - number of epochs to fit each individual

EPOCHS_BEST - how long to train best individual at the end of genetic algorithm

### Genetic algorithm details
ACTIVATION_FUNCTIONS - list of activation function to choose from (supported 'relu', 'lrelu', 'tanh' and 'sigmoid')

LINEAR_FEATURES - list of featurs allowed to be in linear layer during the initialization

USE_CONV - whether to use convolutional layers

CONV_FEATURES - list of featurs allowed to be in convolutional layer during the initialization

KERNEL_SIZES - list of kernel sizes allowed to be in convolutional layer during the initialization

LATENT_SIZE - size of latent space

BEST_MODEL_PATH - path for saving best model during training

### Output configs
ROOT_N_IMAGES - square root of number of images you want to output (for 25 images it is 5, and so on), images will be displayed in a square grid

GENERATED_PATH - path for generated images

TRUE_PATH - path for true images

RECS_PATH - path for recreated images
