import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def plot_training(costs, accs):
    """
    Plots curve of Cost vs epochs and Accuracy vs epochs for 'train' and 'valid' sets during training
    """
    # Find the accuracy
    train_acc = accs['train']
    valid_acc = accs['valid']

    # Find the cost
    train_cost = costs['train']
    valid_cost = costs['valid']

    # Epoch = When entire dataset has been run a single time through the network
    # Find the total number of epochs when accuracy reached a certain level
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, valid_cost)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Cost')

    plt.show()


def n_p(x):
    """convert numpy float to Variable tensor float"""
    return Variable(torch.FloatTensor([x]), requires_grad=False)


# Calculate number of normal/abnormal images for entire category (eg. XR_WRIST)
def get_count(df, cat):
    """
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    """
    # Check if 'positive or negative' in path --> returns 1 or 0
    # Previously when iterating over studies, you'd count every picture in that study
    # Hence, you would do df[df['Path'].str.contains(cat)]['Count'].sum()

    # Don't need to enclose over df[] because we are not calling ['Count']
    return df['Path'].str.contains(cat).sum()


if __name__ == 'main':
    pass
