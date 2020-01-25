###############################################################
# training and evaluating the base CNN trained on augmented 
# HAPPEI datset
###############################################################

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
from skimage import io
from PIL import Image

# Hyperparameters
batch_size = 5
num_epochs = 50
learning_rate = 1e-5 


# creating custom torch compatible dataset
class CDATA(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.mode = train
        self.transform = transform
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

        return image, label

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


# finding weights for each sample based on presence of its class in the dataset 
df = pd.read_csv('./../datasets/augdataset/augmented_data.csv') 
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

train_dataset = CDATA(csv_file='./../datasets/augdataset/augmented_data.csv', root_dir='./../datasets/augdataset/aug_images_new/', train=True, transform=composed_transform)
print('Size of train dataset: %d' % len(train_dataset))

#  Dataloader with upsampleing for smaller classes 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = sampler, pin_memory=True)

test_dataset = CDATA(csv_file='./../datasets/augdataset/Happie_Test.csv', root_dir='./../datasets/Happie/', train=False, transform=composed_transform)
print('Size of test dataset: %d' % len(test_dataset))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

val_dataset = CDATA(csv_file='./../datasets/augdataset/aug_happei_val.csv', root_dir='./../datasets/augdataset/aug_images_new/',train=False, transform=composed_transform)
print('Size of val dataset: %d' % len(val_dataset))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
# Plotting a snippet of training images
train_dataiter = iter(train_loader)
train_images, train_labels = train_dataiter.next()
imshow(torchvision.utils.make_grid(train_images))
plt.show()


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

# Deploying NN on to 4 GPUs
vgg16.features = torch.nn.DataParallel(vgg16.features, device_ids=[0, 1, 2, 3])
vgg16.cuda()

# vgg16.load_state_dict(torch.load('new_weights/vgg_model_100.pth.tar'))

criterion = nn.L1Loss()
optimizer_vgg16 = torch.optim.Adam(vgg16.parameters(), lr=learning_rate,
                                   weight_decay=0.0001)

def loss_function(a, b):
    return (a - b).abs().sum()


def test_vgg16():
    vgg16.eval() 
    correct = 0
    total = 0
    loss = []
    for images, labels in test_loader:
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels.float()).cuda()
        outputs = vgg16(images)
        loss+= [criterion(outputs, labels).data.cpu().numpy()[0]]
    average_loss = sum(loss)/len(loss)
    return average_loss


def train_vgg16():
    vgg16.train()
    total_loss = []
    acc_data = []
    print('Training Model...')

    for epoch in range(num_epochs):
        for i, (images, target) in enumerate(train_loader):

            input_var = torch.autograd.Variable(images).cuda()
            target_var = torch.autograd.Variable(target.float()).cuda()
            output = vgg16(input_var)
            loss = criterion(output, target_var)
            optimizer_vgg16.zero_grad()
            loss.backward()
            optimizer_vgg16.step()
            if (i + 1) % 50 == 0:
                # Compute val set accuracy
                test_loss = test_vgg16()
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Test Mean Absolute Error: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0],test_loss))
                total_loss += [loss.data[0]]

    print('Training Finished')
    return total_loss


Error = train_vgg16()
# # Saving the loss values
with open('train_loss.txt', 'w+') as f:
    pickle.dump(Error, f)


plt.plot(Error) 
plt.xlabel('Iterations')
plt.ylabel('Mean Error on Training Batch')
plt.show()

# Saving the model
torch.save(vgg16.state_dict(), 'new_weights/vgg_model_150.pth.tar')
print('done')

final_mae = test_vgg16()
print('Test mean absolute error of the model on the %d test images: %d %%' % (len(test_dataset), final_mae))

