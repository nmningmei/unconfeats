# Simulations

---
## Experiment settings
```
# folder for saving the trained models
model_dir               = '../models'
# folder that contains the training images
train_folder            = 'greyscaled'
# folder that contains the validation images
valid_folder            = 'experiment_images_greyscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
# to print the progress of the training
print_train             = True #
# resize the images to 128 x 128 x 3
image_resize            = 128
# the batch size for training and validating
batch_size              = 8
# learning rate for the optimizer, here we used ADAM as the optimizer
lr                      = 1e-4
# the maximum of epochs we will train the model
n_epochs                = int(1e3)
# where to train the models
device                  = 'cpu'
# the pretrained model used for building the FCNN, we can change to i.e. alexnet, mobilenet, densenet, resnet50
pretrain_model_name     = 'vgg19_bn'
# the number of hidden units for the hidden layer, i.e. 2, 5, 10, 20, 50, 100, 200, 300
hidden_units            = 5
# the activation function for the hidden layer, i.e. relu, elu, linear, ...
hidden_func_name        = 'selu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
# the dropout rate of the hidden layer, i.e. 0, 0.25, 0.5, 0.75
hidden_dropout          = 0.25
# the maximum consective epochs when the model does not improve on the validation set during training
patience                = 5
# the output activation function, i.e., softmax and sigmoid. It detemines the output layer units
output_activation       = 'softmax'
# name for saving the trained model
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
# am I in the debugging phase?
testing                 = True #
# during the testing phase of the trained FCNN, how many cycles of 96 images will be passed
n_experiment_runs       = 20
# the number of noise levels
n_noise_levels          = 50
# the number of permutations to perform during the statistical inference
n_permutations          = int(1e4)
# we elected to add one example of only noise to the batch to make the FCNN less sensitive to noise
(and here indicate see [Figure]() in the repository showing the drop in FCNN performance from 1 to 0.8 with little noise chance) 
n_noise                 = 1
```

---
## Training
```
train_loop(net, # the model for training - CNN layer + hidden (dense) layer + output (dense) layer
           loss_func, # Binary cross entropy loss
           optimizer, # ADAM with learning rate of 1e-4
           dataloader, # dataloader with a batch size of 8 and simple augmentations
           device, # where to perform the computations, either on GPU or CPUs
           categorical          = True, # determined by the output layer
           idx_epoch            = 1, # determined by the outer loop of the training epochs
           print_train          = False, # whether to print the training progress
           output_activation    = 'softmax', # to call the output activation function
           l2_lambda            = 0, # the L2 proportion
           l1_lambda            = 0, # the L1 proportion
           n_noise              = 0, # number of images that contain only Gaussion noise
           use_hingeloss        = False, # you can use hingloss, but it will require you to modify the output of the model
           )
```
### [Check the binary entropy loss for 0.5 probability labels](https://github.com/nmningmei/unconfeats/blob/main/scripts/simulation/scripts/0.1.binary_cross_entropy_loss.py)

## Validation
```
validation_loop(net,# the model for training - CNN layer + hidden (dense) layer + output (dense) layer
                loss_func,# Binary cross entropy loss
                dataloader,# dataloader with a batch size of 8 and simple augmentations
                device,# where to perform the computations, either on GPU or CPUs
                categorical = True,# determined by the output layer
                output_activation = 'softmax',# to call the output activation function
                verbose = 0,# whether to print the training progress
                )
```
## Testing
```
behavioral_evaluate(net,# the model for training - CNN layer + hidden (dense) layer + output (dense) layer
                    n_experiment_runs,# number of cycles of 96 unique object images that fed to the model
                    loss_func,# Binary cross entropy loss
                    dataloader,# dataloader with a batch size of 8 and simple augmentations
                    device,# where to perform the computations, either on GPU or CPUs
                    categorical = True,# determined by the output layer
                    output_activation = 'softmax',# to call the output activation function
                    verbose = 0,# whether to print the training progress
                    )
```
## Results

### Trained with one example of only noise added to the training batches

### Trained with no example of only noise added to the training batches
