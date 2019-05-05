import torch.nn as nn
import torch.optim
from densenet import densenet169
from utils import n_p, get_count
from train import train_model, get_metrics
from datapipeline import get_study_data, get_dataloaders

if __name__ == '__main__':
    # #### load study level dict data
    study_data = get_study_data(study_type='XR_WRIST')

    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size=1)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # #### Build model
    # tai = total abnormal images, tni = total normal images
    tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    tni = {x: get_count(study_data[x], 'negative') for x in data_cat}

    # Find the weights of abnormal images and normal images
    # Compare get_metrics to this value --> This is the output we are aiming for
    Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

    print('tai:', tai)
    print('tni:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    print('Wt1 valid:', Wt1['valid'])


    class Loss(nn.modules.Module):
        def __init__(self, norm_weight, ab_weight):
            super(Loss, self).__init__()
            self.norm_weight = norm_weight
            self.ab_weight = ab_weight

        def forward(self, inputs, targets, phase):
            loss = - (self.norm_weight[phase] * targets * inputs.log() + self.ab_weight[phase] * (1 - targets) * (
                    1 - inputs).log())
            return loss


    model = densenet169(pretrained=True)
    model = model.cuda()

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    # Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=5)

    # Pytorch automatically converts the model weights into a pickle file
    torch.save(model.state_dict(), 'models/model.pth')

    get_metrics(model, criterion, dataloaders, dataset_sizes)
