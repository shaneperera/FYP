import torch.nn as nn
import torch.optim
from densenet import densenet169
from utils import n_p, get_count
from train import train_model, test_model,train_ensemble
from datapipeline import get_study_data, get_dataloaders
import torch.utils.model_zoo as model_zoo
from resnet import resnet101,resnet152,resnext101,wide_resnet101_2
from inception import inception_v3
from VGG import vgg19_bn
import json
from ensemble import Ensemble,Ensemble2,Ensemble3
from shufflenet import shufflenet_v2_x1_0
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np


if __name__ == '__main__':
    #load JSON file
    with open('Settings.json', 'r') as f:
        settings = json.load(f)

    #selecting run in JSON file
    num_ID = 17
    test = 1

    #load variables from JSON file
    batch_size = settings['run'][num_ID]['bs']
    current_epoch = settings['run'][num_ID]['current_epoch']
    epochs = settings['run'][num_ID]['total_epochs']
    learning_rate = settings['run'][num_ID]['lr']
    droprate = settings['run'][num_ID]['dropout']
    costs = settings['run'][num_ID]['costs']
    accs = settings['run'][num_ID]['accuracy']
    latest_model_path = settings['run'][num_ID]['latest_model_path']
    modeltype = settings['run'][num_ID]['modeltype']


    # #### load study level dict data
    study_data = get_study_data(study_type='XR_WRIST')
    # # print(study_data_wrist)
    # study_data_shoulder =  get_study_data(study_type='XR_SHOULDER')
    # study_data_hand = get_study_data(study_type='XR_HAND')
    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories

    #combining different study types together
    # frames_train = [study_data_wrist['train'],study_data_shoulder['train'],study_data_hand['train']]
    # frames_valid = [study_data_wrist['valid'], study_data_shoulder['valid'], study_data_hand['valid']]
    # study_data = {}
    # study_data['train'] = pd.concat(frames_train)
    # study_data['valid'] = pd.concat(frames_valid)

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


    class Loss(nn.modules.Module):
        def __init__(self, norm_weight, ab_weight):
            super(Loss, self).__init__()
            self.norm_weight = norm_weight
            self.ab_weight = ab_weight

        def forward(self, inputs, targets, phase):
            loss = - (self.norm_weight[phase] * targets * inputs.log() + self.ab_weight[phase] * (1 - targets) * (
                    1 - inputs).log())
            return loss

    if test:
        # model = Ensemble("models/best_model_4.pth","models/best_model_res_12.pth","models/best_model_res_14.pth")
        # model.cuda()
        # criterion = Loss(Wt1, Wt0)
        # test_acc, test_loss = test_model(model,criterion,dataloaders,dataset_sizes)
        # # test_acc= test_model(model, criterion, dataloaders, dataset_sizes)
        # print(test_acc,test_loss)
        # print(test_acc)
        model = densenet169(pretrained=True, droprate=droprate)
        model.cuda()
        model.eval()
        img,_ = (next(iter(dataloaders['valid'])))
        print('img: ',img[0].size())
        inputs = Variable(img[0].cuda())
        pred,features,out_fc = model(inputs)
        print('inputs: ', img[0][0].size())
        print('features: ',features.size())
        print('fc: ', out_fc.size())
        total = torch.sum(out_fc,1)
        print('total :',total)
        out_fc = out_fc.permute(1,0) # might be bad
        out_fc = torch.div(out_fc,total)
        print('fc size: ',out_fc.size())
        out_fc = out_fc.permute(1,0)
        # out_fc = torch.unsqueeze(out_fc,dim=2)
        print('fc size: ', out_fc.size())
        features = features.permute(0,3,2,1)
        print('features: ', features.size())
        heatmap = []
        # torch.mul(out_fc, features)
        for i in range(img[0].size()[0]):
            # print(out_fc[i].size())
            # print(features[i].size())
            # heatmap = torch.mul(out_fc[i],features[i])
            # heatmap1.permute(2,1,0)
            # heatmap.append(heatmap)
            heatmap.append(torch.mul(out_fc[i],features[i]))
            # heatmap[i] = torch.sum(heatmap[i],dim=0)
        print('heatmap: ',heatmap[0].size())
        heatmap[0] = torch.mean(heatmap[0],2)
        # print(heatmap[0])
        # plt.imshow(heatmap[0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(heatmap[1].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(heatmap[2].cpu().detach().numpy())
        # plt.show()
        # print(img[0][0])
        # plt.imshow(img[0][0].permute(1,2,0))
        # plt.show()


        # full_img = cv2.imread('D:/Desktop/FYP/MURA-vtest/valid/XR_WRIST/patient00006/study1_positive/image1')
        full_img = cv2.imread('study1_positive/image1.png')
        # cv2.imshow('image',full_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        heatmap[0] = heatmap[0].cpu().detach().numpy()
        print(heatmap[0])

        heatmap = cv2.resize(heatmap[0],full_img.shape[1], full_img.shape[0])
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('./map.jpg', superimposed_img)


    else:
        if modeltype == "dense":
            model = densenet169(pretrained=True, droprate= droprate)
            # num_features = model.num_features
            # model.classifier = nn.Linear(1664,1)
        elif modeltype == "inception":
            model = inception_v3(pretrained=True)
            # num_ftrs = model.fc.in_features
            model.AuxLogits.fc = nn.Linear(768, 1)
            model.fc = nn.Linear(2048, 1)
        elif modeltype == "vgg":
            model = vgg19_bn(pretrained=True)
            model.classifier[6] = nn.Linear(4096, 1)
        elif modeltype == "ensemble":
            model = Ensemble3()
            ensemble = Ensemble("models/best_model_4.pth","models/best_model_res_12.pth","models/best_model_res_14.pth")
            ensemble.cuda()
        elif modeltype == "shufflenet":
            model = shufflenet_v2_x1_0()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            # model = resnet101(pretrained=True)
            model = resnet101()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)

        if latest_model_path != "":
            model.load_state_dict(torch.load(latest_model_path))
        model.cuda()

        criterion = Loss(Wt1, Wt0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

        # Train model
        # model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=epochs-current_epoch, costs= costs, accs= accs, num_ID = num_ID,modeltype = modeltype)
        model = train_ensemble(model,ensemble,dataloaders,criterion,optimizer,scheduler,dataset_sizes,num_epochs = epochs-current_epoch,costs=costs, accs=accs,num_ID= num_ID)

