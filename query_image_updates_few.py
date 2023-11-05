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
from itertools import combinations
 

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 10)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

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
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM+4) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"),map_location='cuda:0'))
        print("load feature encoder success")
    if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"),map_location='cuda:0'))
        print("load relation network success")

    def get_bb_number(image_name:str):
        image_name=image_name.split('.')
        return int(image_name[0][2:])
    
    def get_frame_number(frame_name:str):
        return int(frame_name[5:])
    
    def IoU(boxA, boxB):
	    # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # return the intersection over union value
            return iou


    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform_ToTensor = transforms.Compose([transforms.ToTensor(),normalize])

    # read image files
    support_folders_path="/content/drive/MyDrive/support/"
    groundtruth_folders_path="/content/drive/MyDrive/groundtruth/"
    comp_image_path="/content/drive/MyDrive/support/frame1/bb11.jpg"
    
    support_folders=os.listdir(support_folders_path)
    support_folders.sort(key=get_frame_number)
#    groundtruth_folders=os.listdir(groundtruth_folders_path)
#    groundtruth_folders.sort(key=get_frame_number)


    f=open('/content/drive/MyDrive/with_update_gt.txt', 'w')

#    query_image_coordinates_int=[789, 488, 789+84, 488+84]
#    query_image_coordinates_str=c.readline()
#    query_image_coordinates_splitted=query_image_coordinates_str.split(",")
#    query_image_coordinates_raw=query_image_coordinates_splitted[1:5]
#    query_image_coordinates_int=[int(x) for x in query_image_coordinates_raw]
    frame_number=0
    query_image=Image.open("/content/drive/MyDrive/support/frame1/bb0.jpg")
    query_image_tensor=transform_ToTensor(query_image).to(torch.float32)

    update_frames=[100,200,300,400,500]

    for support_folder in support_folders:
        bb_names=os.listdir(support_folders_path+support_folder+"/")
        bb_names.sort(key=get_bb_number)
#        groundtruth_image_name=os.listdir(groundtruth_folders_path+support_folder+"/")
#        groundtruth_image_path=groundtruth_folders_path+support_folder+"/"+groundtruth_image_name[0]
#        groundtruth_image=Image.open(groundtruth_image_path)
#        groundtruth_image_tensor=transform_ToTensor(groundtruth_image).to(torch.float32)
#        query_image_coordinates_raw=q.readline()
#        query_image_coordinates_splitted=query_image_coordinates_raw.split(",")
#        query_image_coordinates_int=[int(float(x)) for x in query_image_coordinates_splitted[0:4]]
#        query_image_coordinates=[query_image_coordinates_int[0],
#                                 query_image_coordinates_int[1],
#                                 query_image_coordinates_int[0]+query_image_coordinates_int[2],
#                                 query_image_coordinates_int[1]+query_image_coordinates_int[3]]

        bb_combinations=combinations(bb_names,2)

        frame_relations={}
        
        #for bb in bb_names:
        for pair in bb_combinations:
            tensor_sequence=[]
            support_image=Image.open(support_folders_path+support_folder+"/"+pair[0])
            support_image_tensor=transform_ToTensor(support_image).to(torch.float32)

            tensor_sequence.append(support_image_tensor)
    
            comp_image=Image.open(support_folders_path+support_folder+"/"+pair[1])
            comp_image_tensor=transform_ToTensor(comp_image).to(torch.float32)
            tensor_sequence.append(comp_image_tensor)
            sample_images=torch.stack(tensor_sequence)

        # read line only if first part is 1
        # then hold that line as string
        # split the string by comma into a list
        # obtain bounding box coordinates
        # crop image according to the bounding box number and coordinates
        # transform all cropped images into tensors 
        # hold the number of found bounding boxes to use as shot number  
    #sample_images=torch.stack(images,dim=0).to(torch.float64)

    # calculate features
            #sample_features = feature_encoder(Variable(support_image_tensor.unsqueeze(0)).to(torch.float32).cuda(GPU)) # 5x64
            sample_features = feature_encoder(Variable(sample_images).to(torch.float32).cuda(GPU)) # 5x64
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

            frame_relations[pair[0]+"|"+pair[1]]=relations[0].tolist()[0]
        frame_relations=dict(sorted(frame_relations.items(), key=lambda item: item[1],reverse=True))
        #print(support_folder+" | "+str(frame_relations))
        
        for keys in frame_relations.keys():
            print(support_folder+" | "+str(keys)+" | "+str(frame_relations[keys]))
            f.write(support_folder+" | "+str(keys)+" | "+str(frame_relations[keys])+"\n")

        frame_number+=1

        if frame_number in update_frames:
            query_image=Image.open(support_folders_path+support_folder+"/"+str(list(frame_relations.keys())[0].split("|")[0]))
            print(str(list(frame_relations.keys())[0].split("|")[0]))
            query_image_tensor=transform_ToTensor(query_image).to(torch.float32)
            #update_query=0
            
        
    f.close()

        #tpr=truepositive_number/(truepositive_number+falsenegative_number) #true positive rate
        #acc=(truepositive_number+truenegative_number)/(truepositive_number+truenegative_number+falsepositive_number+falsenegative_number)#accuracy
        





if __name__ == '__main__':
    main()
#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


