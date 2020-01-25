#############################################################
# training and evaluating sCNN
#############################################################

import torch.utils.data 
import torchvision.models as models
import torchvision.transforms as transforms 
import torchvision.utils
import torch.nn as nn
import torch.nn.parallel
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage import io
from PIL import Image
import math
import cv2
from tensorboardX import SummaryWriter
import torch.nn.functional as F


# hyperparameters
batch_size = 5
num_epochs = 5
learning_rate = 1e-5


#loading pretrained model on HAPPIE dataset 
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 512),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
)
vgg16.features = torch.nn.DataParallel(vgg16.features, device_ids=[0])
vgg16.cuda()
vgg16.load_state_dict(torch.load('HAPPIE_CNN_Model/vgg_model_50.pth.tar'))

# sCNN model
class Net(nn.Module):
    def __init__(self, vggnet):
        super(Net, self).__init__()
        self.features = vggnet.features
        self.classifier = vggnet.classifier
        self.gradients = None
    def save_gradient(self, grad):
        self.gradients = grad        
    def forward(self,x):
        x = self.features(x)
        x.register_hook(self.save_gradient)
        y = x.view(x.size(0), -1)
        y = self.classifier(y)
        return [x,y]

model = Net(vgg16)


#data loading 


def img_loader(img):
    img = Image.fromarray(img)
    transform = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transform(img)
    img = Variable(img.float().cuda())
    img = img.unsqueeze(0) # add an additional batch dimension
    return img


class CDATA(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.mode = train
        self.transform = transform
        self.sec_root_dir = 'saliency_gt2' #dir to the saliency images
        # creating mapping between images and labels
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.data.ix[item, 0])
        image = io.imread(img_name)
        image = Image.fromarray(image, mode='RGB')
        label = self.data.ix[item, 1]
        default_CNN_input_size = (224,224)
        image = image.resize(default_CNN_input_size, Image.ANTIALIAS)

        if self.transform is not None:
            image = self.transform(image)
        if self.mode:
            sal_transform = transforms.Compose([transforms.Scale((7,7)),transforms.ToTensor()])
            salient_gt = io.imread(os.path.join(self.sec_root_dir, self.data.ix[item,0]))
            salient_gt = Image.fromarray(salient_gt)
            salient_gt = sal_transform(salient_gt)
            return image, salient_gt, label
        else:
            return image, label

# for adjusting class imbalances in HAPPEI dataset
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight


df = pd.read_csv('./../datasets/augdataset/augmented_data.csv') # csv file having all the image paths and labels
col_headers = list(df)
file_list = df[col_headers[0]].values[:]
label_list = df[col_headers[1]].values[:]

# generating weights
weights = make_weights_for_balanced_classes(label_list, 6)
weights = torch.DoubleTensor(weights)

# creating a biased sampler which will sample based on weights of each class        
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# Transforming images according to the requirement of our network
composed_transform = transforms.Compose([transforms.Scale(224), transforms.ToTensor()])

train_dataset = CDATA(csv_file='./../datasets/augdataset/augmented_data.csv', root_dir='./../datasets/augdataset/aug_images_new/',train=True, transform=composed_transform)
print('Size of train dataset: %d' % len(train_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = sampler, pin_memory=True)

test_dataset = CDATA(csv_file='./../datasets/augdataset/Happie_Test.csv', root_dir='./../datasets/Happie/', train=False, transform=composed_transform)
print('Size of test dataset: %d' % len(test_dataset))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

val_dataset = CDATA(csv_file='./../datasets/augdataset/aug_happei_val.csv', root_dir='./../datasets/augdataset/aug_images_new/',train=False, transform=composed_transform)
print('Size of val dataset: %d' % len(val_dataset))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True)


# optimizer and losses
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=0.0001)
criterion = nn.L1Loss()


def saliency_loss(output, gt, grads):
    no_samples = output.size()[0]
    ws = grads.view((no_samples,512,-1)).mean(2)
    ws.unsqueeze_(-1)
    ws = ws.expand(no_samples,512,7)
    ws.unsqueeze_(-1)
    ws = ws.expand(no_samples,512,7,7)
    ws = Variable(torch.from_numpy(ws.cpu().data.numpy())).cuda()
    foutput = ws.mul(output)
    foutput = foutput.sum(1)
    foutput = F.relu(foutput)
    min_value = Variable(torch.from_numpy(foutput.min().cpu().data.numpy())).cuda()
    max_value = Variable(torch.from_numpy(foutput.max().cpu().data.numpy())).cuda()
    foutput = (foutput - min_value)/(max_value - min_value)
    loss = foutput.sub(gt)**2
    new_loss = loss.squeeze()
    new_loss = new_loss.sum(0)
    # rescaling saliency loss to balance between saliency and score prediction loss
    return 1e-3*new_loss


def test():
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    loss = []
    for images, labels in test_loader:
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels.float()).cuda()
        _ ,outputs = model(images)
        loss+= [criterion(outputs, labels).data.cpu().numpy()[0]]
    average_loss = sum(loss)/len(loss)

    return average_loss

def val():
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    loss = []
    for images, labels in val_loader:
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels.float()).cuda()
        _ ,outputs = model(images)
        loss+= [criterion(outputs, labels).data.cpu().numpy()[0]]
    average_loss = sum(loss)/len(loss)

    return average_loss

def train():
    # switch to train mode
    model.train()
    global train_loss
    global val_mae
    global test_mae 
    print('Training Model...')

    for epoch in range(num_epochs):
        for i, (images, salient_gt, target) in enumerate(train_loader):

            input_var = torch.autograd.Variable(images).cuda()
            target_var = torch.autograd.Variable(target.float()).cuda()
            salient_gt = torch.autograd.Variable(salient_gt).cuda()
            saliencies, scores = model(input_var)
            closs = criterion(scores,target_var)
            optimizer.zero_grad()
            closs.backward(retain_graph=True)
            # backpropagation for saliency
            gradients = model.gradients
            sloss = saliency_loss(saliencies, salient_gt, gradients)
            sloss.backward(torch.ones((7,7)).cuda(), retain_graph=True)
            optimizer.step()
            
            if (i + 1) % 200 == 0:
                # Compute test set accuracy
                test_loss = test()
                val_loss = val()
                # For real time plot
                ax.clear()
                ax.plot(range(len(val_mae)), val_mae, range(len(test_mae)), test_mae)
                fig.canvas.draw()
                
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Val MAE: %.4f Test Mean Absolute Error: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, closs.data[0], val_loss, test_loss))
                train_loss += [closs.data[0]]
                val_mae += [val_loss]
                test_mae += [test_loss]
 
    print('Training Finished')

# For generating real time plot
fig = plt.figure()
ax = fig.add_subplot(111)

fig.show()
fig.canvas.draw()
train()