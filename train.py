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

epochs=50
batch_size=32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*2*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(np.unique(ytrain)))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    folders=glob('English/Fnt/*')
    files=[]
    for folder in folders:
        files=files+glob(folder+'/*.png')

    X=np.zeros((len(files),1,20,20))
    y=np.zeros(len(files)).astype('int')

    for i in range(len(files)):
        temp_img=cv2.imread(files[i],0)
        y[i]=int(files[i].split('\\')[2].split('-')[0][3:])-1 #Windows

        #y[i]=int(files[i].split('/')[3].split('-')[0][3:])-1 #Linux
        temp_img=cv2.resize(temp_img,(20,20))
        temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        X[i,0,:,:]=255-np.copy(temp_img)

    X=X/X.max()
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    prev_val_acc=0
    for epoch in range(epochs):
        running_loss = 0.0
        rand_ids=np.arange(Xtrain.shape[0])
        shuffle(rand_ids)
        Xtrain=Xtrain[rand_ids]
        ytrain=ytrain[rand_ids]
        for i in range(int(Xtrain.shape[0]/batch_size)):
            
            inputs=torch.from_numpy(Xtrain[i*batch_size:(i+1)*batch_size]).float()
            labels=torch.from_numpy(ytrain[i*batch_size:(i+1)*batch_size])
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                inputs=torch.from_numpy(Xtrain).float()
                inputs=inputs.to(device)
                outputs=net(inputs)
                outputs=np.argmax(outputs.detach().cpu().numpy(),axis=1)
                print('Train accuracy:',100*(outputs==ytrain).sum()/ytrain.shape[0])


                inputs=torch.from_numpy(Xtest).float()
                inputs=inputs.to(device)
                outputs=net(inputs)
                outputs=np.argmax(outputs.detach().cpu().numpy(),axis=1)
                current_acc=100*(outputs==ytest).sum()/ytest.shape[0]
                print('Validation accuracy:',current_acc)
                if current_acc>prev_val_acc:
                    torch.save(net,'net.pt')
                prev_val_acc=current_acc