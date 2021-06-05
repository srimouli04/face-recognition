# face-recognition

## Please follow the below steps to execute my scripts from docker bash :

##### Datasets and Trained models are available here https://drive.google.com/drive/folders/1mv3yeKsUygBrmvhIFI-KEEbB7_eFIE0_?usp=sharing

1. docker pull srimoulib/skylark-intern-submission:v2.2

2. docker run -it -v <Dataset Path from your computer>:/intern/dataset srimoulib/skylark-intern-submission:v2.2 bash  

3. Once the bash is loaded please use the command "python3 reorganise_dataset.py --type TTV" this command splits the dataset into the required Test, Train and Validation datasets and creates a new directory FRDataset which is used by my scripts to predict. This is essential as the class_names are picked from these directory names. 
   The below is the structure of FRDataset directory. The reorganise_dataset splits data in the master dataset in ration of 80:20 split to train and test dataset. The validation dataset is prepared from taking 20% of images from train dataset.

# FRDataset

    ## Train
      → chris_evans
      → chris_hemsworth
      → mark_ruffalo
      → robert_downey_jr
      → scarlett_johansson
    ## Test
      → chris_evans
      → chris_hemsworth
      → mark_ruffalo
      → robert_downey_jr
      → scarlett_johansson
    ## Validation
      → chris_evans
      → chris_hemsworth
      → mark_ruffalo
      → robert_downey_jr
      → scarlett_johansson
  

4. I have trained the models for both GPU and CPU. The same can be found at "Trained_models" folder. If the "train.py" script is run the models get replaced.

5.In case any new images need to be tested with trained model you may upload the image in the directory referenced to docker volume and pass the directory path as argument to test.py file with Mode 'I' and it could give the prediction. 
"python3 test.py --Mode I --image_path 'New_images/1.jpg' " -> There is one sample image for test in New_images folder but the command can utilise new images also when uploaded via directory referenced to docker volume
"python3 test.py --Mode I --image_path 'dataset/1.jpg' " . If no command is passed then the test evaluation is performed on all images present in FRDataset/Test
