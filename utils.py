import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt



def plot_training(costs, accs, num_ID):
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

    #save plot in plots folder
    plt.savefig('plots/run_'+ str(num_ID) +'.png',bbox_inches='tight')
    plt.show()


def n_p(x):
    """convert numpy float to Variable tensor float"""
    return Variable(torch.FloatTensor([x]), requires_grad=False)


def get_count(df, cat):
    """
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    """
    # By looking at the path column in the pandas dataframe --> Check for the keywords (positive or negative)
    return df[df['Path'].str.contains(cat)]['Count'].sum()


if __name__ == 'main':
    pass
