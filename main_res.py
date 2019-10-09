import torch.nn as nn
import torch.optim
from utils import n_p, get_count
from train import train_model
from datapipeline import get_study_data, get_dataloaders
from torchvision.models import resnet101
import torch.utils.model_zoo as model_zoo
import json

if __name__ == '__main__':
    #load JSON file
    with open('Settings.json', 'r') as f:
        settings = json.load(f)

    #selecting run in JSON file
    num_ID = 0

    #load variables from JSON file
    batch_size = settings['run'][num_ID]['bs']
    current_epoch = settings['run'][num_ID]['current_epoch']
    epochs = settings['run'][num_ID]['total_epochs']
    learning_rate = settings['run'][num_ID]['lr']
    droprate = settings['run'][num_ID]['dropout']
    costs = settings['run'][num_ID]['costs']
    accs = settings['run'][num_ID]['accuracy']
    latest_model_path = settings['run'][num_ID]['latest_model_path']
    modeltype = settings['run'][num_ID]['modetype']

    # #### load study level dict data
    study_data = get_study_data(study_type='XR_WRIST')
    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size)
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

    #ResNet model
    modeltype = 'res'
    model = resnet101(pretrained=True)
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'), strict=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    if latest_model_path != "":
        model.load_state_dict(torch.load(latest_model_path))
    model = model.cuda()
    #cudnn.benchmark = True

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    # Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=epochs-current_epoch, costs= costs, accs= accs, num_ID = num_ID,modeltype = modeltype)

