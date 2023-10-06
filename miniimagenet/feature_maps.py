#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
 

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        

'''
def main():

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"),map_location='cuda:0'))
        print("load feature encoder success")
    if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"),map_location='cuda:0'))
        print("load relation network success")

    
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform_ToTensor = transforms.Compose([transforms.ToTensor(),normalize])

    support_image_path="/content/drive/MyDrive/support/frame1/bb0.jpg"
    comp_image_path="/content/drive/MyDrive/support/frame1/bb1.jpg"
    query_image_path="/content/drive/MyDrive/groundtruth/frame1/bb0.jpg"
        
    query_image=Image.open(query_image_path)
    query_image_tensor=transform_ToTensor(query_image).to(torch.float32)
    
    tensor_sequence=[]
    support_image=Image.open(support_image_path)
    support_image_tensor=transform_ToTensor(support_image).to(torch.float32)
    
    tensor_sequence.append(support_image_tensor)
    for i in range(4):
        comp_image=Image.open(comp_image_path)
        comp_image_tensor=transform_ToTensor(comp_image).to(torch.float32)
        tensor_sequence.append(comp_image_tensor)
    sample_images=torch.stack(tensor_sequence)
    

    # calculate features
    sample_features = feature_encoder(Variable(sample_images).to(torch.float32).cuda(GPU)) # 5x64
    print(sample_features.shape)
    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
    sample_features = torch.sum(sample_features,1).squeeze(1)
    test_features = feature_encoder(Variable(query_image_tensor.unsqueeze(0).to(torch.float32)).cuda(GPU)) # 20x64

    # calculate relations
    # each batch sample link to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    sample_features_ext = sample_features.unsqueeze(0)

    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
    test_features_ext = torch.transpose(test_features_ext,0,1)
    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

    print(feature_encoder)
    
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    model_children=list(feature_encoder.children())
    print(len(model_children))
    print(model_children[0][0])

    for i in range(len(model_children)):
        print(model_children[i][0])
        model_weights.append(model_children[i][0].weight)
        conv_layers.append(model_children[i][0])

    outputs = [conv_layers[0](sample_images.cuda(GPU))]#Holds feature maps
    names = []
    print(support_image_tensor.unsqueeze(0).shape)#.cuda(GPU)

    for i in range(1,len(conv_layers)):
        outputs.append(conv_layers[i](outputs[-1]))
    print(len(outputs))
#print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)
    #counter to keep count of the conv layers
    #append all the conv layers and their respective wights to the list
    
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    
    for num_layer in range(len(outputs)):
      layer_viz = outputs[num_layer].tolist()
      print("Layer ",num_layer+1)
      for filters in layer_viz:
        for filter in filters:
            Image.save("Feature_Map")
            
    


if __name__ == '__main__':
    main()
'''