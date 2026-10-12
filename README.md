# GenAI_density_urbanization
This repository contains code for generating urban density images using generative artificial intelligence (GenAI). The workflow includes dataset preparation, training, validation, and optional customization for different datasets. 

## 1. Setup Instructions
Run the following commands to set up the environment:
```bash
conda env create -f environment.yaml
conda activate cnet_gud
```

## 2. Dataset Preparation
download dataset from [Google Drive](https://drive.google.com/drive/folders/1abhfipdLoHHeEN9F-js9GaZy3a9hfI44?usp=drive_link). 
Place the dataset under the directory:
```bash
./urban_data
```

## 3. Training the Model
Run the training code using:
```bash
python train_density.py
```
Notes:
- The code will automatically save model weights every 5 epochs.
- Training will not stop automatically. You can manually stop it when satisfactory results are achieved (recommended: 20–50 epochs).
- Training progress and model weights are saved under the ./models/ directory.

## 4. Validating Results
Specify the epochs you want to test on in results_view_density.py:
```bash
train_No = 'density'
ckpt_directory = f'./ckpts_s/checkpoints_{train_No}'
ckpt_epoch_list = ['19', '31'] 
```
Run the validation code to generate images using the trained model:
```bash
python results_view_density.py
```
Notes:
- The code generates images for the test dataset.
- Results will be saved to the ./output_image directory.

## 5. Using a Custom Dataset
To use your own dataset, follow these steps:
### A. Update Data Paths
Modify the file paths in train_density.py. For example:
```bash
hint_dir = './urban_data/Chicago/hint_images/' 
image_dir = './urban_data/Chicago/satellite_images/'
desc_dir = './urban_data/Chicago/descriptions/' 
```
### B. Customize Prompt Descriptions
Update the description logic in satellite_tiles_density.py. For example:
```bash
def compute_description(self, row):
  return f"Satellite Image of YourCity. Total built-up volume is {int(row['built_volume_total']/1000)} thousand m3. Non-residential built-up volume is {int(row['built_volume_nres']/1000)} thousand m3."
```
### C. Customize Image Data
Update the data loading paths in satellite_tiles_density.py. For example:
```bash
def __getitem__(self, item):
  ...
  if city == "Chicago":
      hint_name = self.hint_dir + city + '_of_Stage_1_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.tif'
      img_name =  self.image_dir + city + '_of_Stage_4_grid_' + r + '_' + d + '/' + str(xtile) + '_' + str(ytile) + '_' + r + '_' + d + '.jpg'
```
### D. For validation, customize dataset and prompt description in results_view_density.py.
