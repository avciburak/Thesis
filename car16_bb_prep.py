from PIL import Image
import os

frames_path="/content/drive/MyDrive/VOT/sequences/car16/color/"
frame_folders_path="/content/drive/MyDrive/car16/"
bb_path="/content/drive/MyDrive/car16_bbs.txt"

for frame_number in range(1,1993):
    os.mkdir(frame_folders_path+"frame"+str(frame_number))

def get_image_number(image_name:str):
    image_name=image_name.split('.')
    return int(image_name[0])
    
def get_frame_number(frame_name:str):
    return int(frame_name[5:])
 
frames=os.listdir(frames_path)
frames.sort(key=get_image_number)#sorts frames by numbers. 00000001.jpg,00000002.jpg,...

all_bbs=[]

with open(bb_path) as f:#stores all bounding box coordinates according to frames
    for line in f:
        bb=[]
        splitted_line=line.split(",")
        bb.append(int(splitted_line[0])) 
        bb.append(int(splitted_line[1]))
        bb.append(int(splitted_line[2]))
        bb.append(int(splitted_line[1])+int(splitted_line[3]))
        bb.append(int(splitted_line[2])+int(splitted_line[4]))
        all_bbs.append(bb)

bb_number=0
for frame in frames:
    Image.open(frames_path+frame)
    for bb in all_bbs:
        if bb[0]==int(frame.split(".")[0]):
            Image.crop(bb[1],bb[2],bb[3],bb[4])
            Image.save("bb"+str(bb_number)+".jpg")
        bb_number+=1

    