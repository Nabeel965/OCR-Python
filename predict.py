import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from random import shuffle
from matplotlib.patches import Rectangle


lookup_arr=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
            'H','I','J','K','L','M', 'N','O','P','Q','R','S','T','U','V','W','X',
            'Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                                    'p','q','r','s','t','u','v','w','x','y','z']

classes=len(lookup_arr)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*2*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_cc(img):
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    kernel = np.ones((2,2),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    img= cv2.dilate(img,kernel,iterations = 1)

    num_labels, labels_im,stats,centroids = cv2.connectedComponentsWithStats(img,connectivity=4)

    label_hue = np.uint8(254*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.figure(figsize=(15,15))

    plt.imshow(labeled_img)
    plt.savefig('results/connected_component.png')

    return img, stats

def compute_edges(stats,img_shape):

    min_val=stats[1:,4].mean()-4*np.sqrt(stats[1:,4].std())
    max_val=stats[1:,4].mean()+4*np.sqrt(stats[1:,4].std())
    row_vals=np.zeros(img_shape)
    for i in range(stats.shape[0]):
        if stats[i,4]>min_val and stats[i,4]<max_val:
            for j in range(int(stats[i,3])):
                row_vals[int(stats[i,1])+j]=row_vals[int(stats[i,1])+j]+1
    row_vals[row_vals<5]=0
    edges=[]
    for i in range(2,row_vals.shape[0]):
        if row_vals[i]==0 and not(row_vals[i-1]==0):
            edges.append(i-1+5)
        if row_vals[i-1]==0 and not(row_vals[i]==0):
            edges.append(i-10)

    return edges,min_val,max_val


def predict_chars(net,stats,edges,min_val,max_val):

    text_poses=[]
    plt.figure(figsize=(20,20))
    plt.imshow(255-img,cmap='gray')
    pred_arr=np.zeros(stats.shape[0])
    for i in range(1,stats.shape[0]):
        temp=img[stats[i,1]:stats[i,1]+stats[i,3],stats[i,0]:stats[i,0]+stats[i,2]]
        temp=cv2.resize(temp,(16,16))
        new_temp=np.zeros((20,20))
        new_temp[2:18,2:18]=np.copy(temp)
        temp=np.copy(new_temp)

        temp=temp/temp.max()

        pred=net(torch.from_numpy(temp).unsqueeze(0).unsqueeze(0).to(device).float()).detach().cpu().numpy()
        pred_class=np.argmax(pred,axis=1)

        pred_score=np.max(pred)
        if stats[i,4]>min_val and stats[i,4]<max_val and np.log(pred_score)>0:

            x_val= stats[i,1]

            if len(np.where(edges<x_val)[0])==0:
                x_val_ind=0
            else:
                x_val_ind=np.where(edges<x_val)[0][-1]+1

            text_poses.append([1+ x_val_ind,stats[i,0],stats[i,1],stats[i,2],stats[i,3],int(pred_class)])
            plt.gca().add_patch(Rectangle((stats[i,0],stats[i,1]),stats[i,2],stats[i,3],linewidth=1,edgecolor='g',facecolor='none'))
            plt.text(stats[i,0], stats[i,1],lookup_arr[int(pred_class)] )
    plt.savefig('results/characters.png')


    return text_poses


def predict_text(text_poses):
    text_poses=np.array(text_poses)
    for row in np.unique(text_poses[:,0]):
        line=''
        row_poses=text_poses[text_poses[:,0]==row]
        row_poses=row_poses[row_poses[:,1].argsort()]
        prev_point=row_poses[0,1]
        for j in range(row_poses.shape[0]):
            if (row_poses[j,1]-prev_point)>10:
                line=line+' '

            line=line+lookup_arr[row_poses[j,5]]
            prev_point=row_poses[j,1]+row_poses[j,3]
        print(line)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    net=torch.load('net.pt')

    img=cv2.imread('20000-leagues-006-2.jpg',0)
    img=255-img
    img,stats=compute_cc(img)
    edges,min_val,max_val=compute_edges(stats,img.shape[0])
    text_poses=predict_chars(net,stats,edges,min_val,max_val)
    predict_text(text_poses)


