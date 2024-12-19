# F24-ML-Final-Project
Group: Johnny Pham, Justin Wang, Zijiang Yang

Dataset included in this repo.

Total Project Steps:
1. Gather Data into a folder with 4 subfolders titled Catfih, Snakehead, Bluegill and Bass (Our code used 100 fish per class)
2. Run Segmentation of each of the folders with Fish Segmenter RNN and hand segment any failed segmentations
3. Run the Data Augment Notebook and change the paths to work accordingly with your folder structure
4. Run the the FishClassifierResNet Notebook and change the paths to work accordingly with your folder structure

## Folder Path Structure
FinalMLProject
    - all_fish
        - Bass
            - Mask
            - Premask
        - Bluegill
            - Mask
            - Premask
        - Catfish
            - Mask
            - Premask
        - Snakehead
            - Mask
            - Premask
    - data
        - Bass ... (same 4 fish class as before, not included for brevity)
    - split_data
        - test
            - Bass ...
        - train
            - Bass ...
    - aug_data
        - train
            - Bass ...

Fish Segmenter Steps:
*Run pip install -r requirement.txxt to install all the required libraries
*In the mask_r_cnn.py code, there is a line for an input folder and an output folder (change the path to match those folders)
*Create an input and an output folder if they do not exist
*Run the the python file
*The segmented pictures will be in the output folder

Data Augmenting Steps:
* Specify Base, Data, Split_data, test and train directory.
* Run Notebook

Fish Classifier Resnet Steps:
* Specify the Base Directory that contains the Final Project Folder
* Specify the Train and Test Directory
* Run the rest of the notebook preferrable with GPU
* View Results
