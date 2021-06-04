# -*- coding: utf-8 -*-
"""
@author: Srimouli Borusu
@Email : srimouli04@yahoo.co.in 
"""

import shutil
import os 
import argparse

directory = '/dataset/'
data_dir = 'FRDataset'

def parse_args():
  parser = argparse.ArgumentParser(description='Build Test Data if you want to run the Test.py standalone')
  parser.add_argument('--data_dir', default='dataset', help='training set directory')
  parser.add_argument('--type', default='', help='which directory do you want to build? TTV -> Training, Test, Validation or T only Test')
  
  args = parser.parse_args()
  return args

args = parse_args()

directory = args.data_dir

def buildTrainValTestData():
  if not os.path.isdir('FRDataset'):
    os.mkdir('FRDataset')
    os.mkdir('FRDataset/train')
    os.mkdir('FRDataset/validation')
    os.mkdir('FRDataset/test')
  if not os.path.isdir('FRDataset/train'):
    os.mkdir('FRDataset/train')
  if not os.path.isdir('FRDataset/validation'):
    os.mkdir('FRDataset/validation')
  if not os.path.isdir('FRDataset/test'):
    os.mkdir('FRDataset/test')

def buildTestData():
  if not os.path.isdir('FRDataset'):
    os.mkdir('FRDataset')  
  if not os.path.isdir('FRDataset/test'):
    os.mkdir('FRDataset/test')

if args.type == '':
    try:
         shutil.rmtree('FRDataset', ignore_errors = False)
    except:
         print("Please create the Driverfile FRDataset by using below command: \n")
         print("python3 reorganise_dataset.py --type TTV ")

if args.type == 'TTV':
         
    buildTrainValTestData()
    
    for file in os.listdir(directory):
        path=os.path.join(directory,file)
        if not os.path.isdir('FRDataset/train/'+file):
          os.mkdir('FRDataset/train/'+file)
        if not os.path.isdir('FRDataset/validation/'+file):
          os.mkdir('FRDataset/validation/'+file)
        if not os.path.isdir('FRDataset/test/'+file):
          os.mkdir('FRDataset/test/'+file)
    
        Tot_files = len(os.listdir(path))
        No_of_files_in_train = round(Tot_files * 0.8)
        No_of_files_in_Validation = round(No_of_files_in_train * 0.15)
        t = 1
        v = 1
        for im in os.listdir(path):
          if t <= No_of_files_in_train:
            shutil.copy(os.path.join(directory,file,im), 'FRDataset/train/'+file)
            t += 1
          else:
             shutil.copy(os.path.join(directory,file,im), 'FRDataset/test/'+file) 
        for im in os.listdir('FRDataset/train/'+file):
          if v <= No_of_files_in_Validation:
            shutil.move(os.path.join('FRDataset/train/'+file,im), 'FRDataset/validation/'+file)
            v += 1

#This eats away the memory with args.type == 'T' and you run into errors      
if args.type == 'T':
    print('This mode can cause errors due to memory limitations')
    buildTestData()
    
    for file in os.listdir(directory):
        path=os.path.join(directory,file)
        if not os.path.isdir('FRDataset/test/'+file):
          os.mkdir('FRDataset/test/'+file)
    
        for im in os.listdir(path):
             shutil.copy(os.path.join(directory,file,im), 'FRDataset/test/'+file) 
