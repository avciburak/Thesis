from PIL import Image
import os
import torchvision.transforms as transforms
import torch

transform_PILToTensor = transforms.Compose([transforms.PILToTensor()])
images=[]
images_dir="C:/Users/vatan/OneDrive/Masaüstü/miniimagenet_images"
image_path_list=[]
for image_paths in os.listdir(images_dir):
    image_path_list.append(os.path.join(images_dir,image_paths))
for image in image_path_list:
    images.append(transform_PILToTensor(Image.open(image)))
stacked=torch.stack(images,dim=0)
print(stacked.size())
print(stacked.to(torch.float64))

