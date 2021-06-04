# -*- coding: utf-8 -*-
"""
@author: Srimouli Borusu
@Email : srimouli04@yahoo.co.in 
"""
import torch
import torch.nn.functional as F
import os 
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
from PIL import Image
import numpy as np


''' functions'''
def parse_args():
  parser = argparse.ArgumentParser(description='Please enter mode of operation')
  parser.add_argument('--Mode', default='D', help='Mode D-Entire test dataset would be evaluated and accuracy is calculated, Mode - I individual image is evaluated and prediction is displayed ')
  parser.add_argument('--image_path', default='', help='Path of individual image')
  args = parser.parse_args()
  return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def predict_individual_image(image_path, model, topk=5):
    
    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)


args = parse_args()

if torch.cuda.is_available():
  model = torch.load('Trained_Models/mod-gpu.model')
else:
  model = torch.load('Trained_Models/mod-cpu.model')

data_transforms_test = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
}
data_dir = 'FRDataset'

# Load the datasets with ImageFolder
image_datasets_test = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms_test[x]) for x in ['test']}

# Using the image datasets and the trainforms, define the dataloaders
batch_size = 64
dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets_test[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['test']}

class_names = image_datasets_test['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.Mode == 'I':
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    image_path = args.image_path
    img = Image.open(image_path)
    threshold = 35 
    probs, classes = predict_individual_image(image_path, model.to(device))
    if round(max(probs)*100) > threshold:
      print(idx_to_class[classes[0]], 'Confidence->',round(probs[0]*100),'%')
    else:
      print('unknown')         
else:
    model.eval()
    accuracy = 0
    phase = 'test'
    for inputs, labels in dataloaders_test['test']:
      inputs = inputs.to(device)
      labels = labels.to(device)
    
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      equality = (labels.data == outputs.max(1)[1])
      
    # statistics
    
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    print('Accuracy on {} data ->: {:.4f}'.format(phase, accuracy))