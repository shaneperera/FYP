import time
import copy
import torch
from torchnet import meter
from torch.autograd import Variable
from utils import plot_training

data_cat = ['train', 'valid']  # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler,
                dataset_sizes, count, num_epochs):
    # In order to determine how long each epoch takes to travel in the network,
    # measure the time since the beginning of the first epoch
    since = time.time()

    # Create a deepcopy of the model weights --> If there is a better model, it will save those weights
    # Creates a replica of the weights --> If new weights lead to lower loss --> Save that instantiation of the network
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialise the accuracy
    best_acc = 0.0

    # Store the cost & accuracy per epoch
    # Each is a dictionary with keys valid and train --> List is defined to store the values for each epoch
    costs = {x: [] for x in data_cat}  # for storing costs per epoch
    accs = {x: [] for x in data_cat}  # for storing accuracies per epoch

    # Print the number of dataloaders present (each dataloader has specific batch of images)
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')

    # For each iteration of an epoch
    # 1. Passes the images through the model and calculates the loss.
    # 2. Backpropagates during the training phase. For the validation/testing phase, it
    # does not adjust the weights.
    # 3. The loss is accumulated across batches for each epoch.
    # 4. The best model is stored and validation accuracy is printed.
    for epoch in range(num_epochs):
        # Single label classification problems (+ve or -ve)
        # Confusion matrix is used to determine the performance of the model
        # meter.ConfusionMeter(k,normalized)
        # k = indicates the number of classes in the classification problem (in this case we have 2 --> +ve or -ve)
        # Normalised = Contains normalised values (like %) if not (counts)
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True) for x in data_cat}

        # Print what epoch you are up to (Eg. Epoch 1/25)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # Print a spacing to make it look cleaner (Eg. -------------------)
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in data_cat:
            # Scheduler step --> Sets the learning rate of each parameter group to the initial learning rate
            # decayed by gamma every step_size epochs
            # Eg. Gamma = 0.1, step size = 10, Initial LR = 0.01
            # Every 10 epochs --> LR = 0.01 * (0.1)^N
            # Second iteration of 10 Epochs, the learning rate is: 0.01*0.1 = 0.001
            # Learning rate controls how much we are adjusting the weights of our network with respect to the loss
            # gradient
            # NOTE: Smaller learning rates mean that you take a longer time to converge --> But you don't miss local
            # minima in your loss function --> Remember you want to find the local minimum as fast as possible
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0
            study_count = count[phase] # Number of images per study (you need to find max of this to pad to this value)
            k = 0 #iterater for the count array
            # Iterate over data
            # Enumerate --> Loop over something and have an automatic counter
            # Eg. Enumerate(dataloaders['train'],2) --> Start at the second index and begin counting

            for i, data in enumerate(dataloaders[phase]):
                # Print the iteration ( '\r' --> Overwrite the existing iteration each time)
                print(i, end='\r')

                for j, study in enumerate(data[0]):
                    # Class ImageDataset returns sample, which is a dictionary that has keys 'images' and 'labels'
                    # 'images' --> Stores the transformed images (there can be multiple from each study)
                    # Start indexing from the first image
                    # index 0 looks into the study of the batch
                    # inputs = data['images'][j]
                    inputs = study #[0:study_count[k]-1]
                    #k += 1
                    # Convert the label (0 or 1) to an integer Tensor
                    labels = data[1][j].type(torch.Tensor)


                    # Wrap them in Variables
                    # NOTE: A variable forms a thin wrapper around a tensor object, its gradients,
                    # and a reference to the function that created it.
                    # The loss function should give a scalar value --> Optimiser will use scalar value and determine
                    # next epoch's ideal weights
                    # print('labels pre', labels.shape)
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # zero the parameter gradients --> Why? Back propagation accumulates gradients, and you don't want
                    # to mix up gradients between mini batches
                    optimizer.zero_grad()

                    # Forward propagation (find output)
                    # When you feed the images into the model, it will yield a probability of the classification
                    outputs = model(inputs)
                    # Find the average of the probability of the classification
                    # We comment it out because we need tensor for torch.max operation
                    outputs = torch.mean(outputs)
                    # Calculate the LOSS (Error) of the classification
                    # Creates a criterion that measures the mean absolute error (MAE) between each element in
                    # the output and target (labels) based on whether it is for train or validation
                    loss = criterion(outputs, labels, phase)
                    running_loss += loss.data[0]

                    # Why do we back propagate here? We want to recreate the image to determine the spatial frequency
                    # features
                    # backward propagation + optimize only if in training phase
                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                    # NOTE: Variable is a wrapper and has multiple components --> We only need to access the data component
                    # Use .detach to access the data for Variables
                    # Preds = Prediction of the classification --> Will be between 0 and 1 --> However we want it above 0.5
                    # Convert the tensor from CPU to GPU to decrease run time
                    # Statistics
                    # Outputs is a tensor (array) --> There should only be a single value
                    # preds = torch.max(outputs.data, 1)
                    preds = (outputs > 0.5).type(torch.cuda.FloatTensor)
                    running_corrects += torch.sum(torch.eq(preds, labels.data))
                    #confusion_matrix[phase].add(preds, labels.data)

                    # print('inputs[0]:', inputs.shape)
                    # print('inputs:', data['images'].shape)
                    # print('labels:', labels.shape)
                    # print('labels.data:', labels.data.shape)
                    # print('labels.data[0]:', labels.data.shape)
                    # print('outputs:', outputs.shape)
                    # print('preds', preds)
                    # print('labels.data', labels.data)
                    # print('dataset_size', dataset_sizes)
                    # print('eq', torch.sum(torch.eq(preds, labels.data)))
                    # print('run correct', running_corrects)

            # Calculate the loss and accuracy
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            # print('acc', epoch_acc)
            # print('acc2', epoch_acc.item())
            # print('acc3', running_corrects.item()/dataset_sizes[phase]) this one works

            # Append onto the empty lists
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            # Print the loss & Accuracy for each epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            #print('Confusion Meter:\n', confusion_matrix[phase].value())

            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                # If the accuracy of current epoch is better than previous epoch, make a replica of the weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # Determine the time it has taken to run the epoch
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    # All the epochs have completed (training phase complete) -> This is how long it took the training phase to complete
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Print the best accuracy
    print('Best valid Acc: {:4f}'.format(best_acc))

    # Plot the costs and accuracy vs epoch
    # NOTE: costs and accs is a dictionary with keys 'valid' and 'train' --> I have defined a list for each of them
    plot_training(costs, accs)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Up until now, we have only calculated the loss and accuracy for each epoch individually for either valid and train
# Now find the total acc, loss and confusion meter for ALL the epochs in the valid and train set
def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    """
    Loops over phase (train or valid) set to determine acc, loss and
    confusion meter of the model.
    """
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        # data is a dictionary with keys 'label' and 'images' --> Check output of ImageDataset class
        labels = data[1].type(torch.Tensor)
        inputs = data[0][0]
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # forward propagation
        # Find the prediction of the classification

        # Outputs a tensor corresponding to each epoch
        outputs = model(inputs)
        # outputs = torch.mean(outputs)
        loss = criterion(outputs, labels, phase)
        # statistics

        running_loss += loss.data[0]

        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        # preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(torch.eq(preds,labels.data))
        #confusion_matrix.add(preds, labels.data)

    loss = running_loss.item() / dataset_sizes[phase]
    acc = running_corrects.item() / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    #print('Confusion Meter:\n', confusion_matrix.value())
