# SKIN CANCER CLASSIFICATION DEEP LEARNING WITH CNN MODEL
Skin cancer is the most common human malignancy, is primarily diagnosed visually, beginning with an initial clinical screening and followed potentially by dermoscopic analysis, a biopsy and histopathological examination. Automated classification of skin lesions using images is a challenging task owing to the fine-grained variability in the appearance of skin lesions.

This the HAM10000 ("Human Against Machine with 10000 training images") dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts.

It has 7 different classes of skin cancer which are listed below :
1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

In this kernel I will try to detect 7 different classes of skin cancer using Convolution Neural Network with keras tensorflow in backend and then analyse the result to see how the model can be useful in practical scenario.
We will move step by step process to classify 7 classes of cancer.

# Step 1 : Install Kaggle Extension and Download Dataset
in this step i will install extension Kaggle on google colab because i need to download the dataset from Kaggle datasets.
```python
! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
Downloading skin-cancer-mnist-ham10000.zip to /content
100% 5.20G/5.20G [01:49<00:00, 20.7MB/s]
100% 5.20G/5.20G [01:49<00:00, 50.9MB/s]
! mkdir skin_cancer
! unzip skin-cancer-mnist-ham10000.zip -d skin_cancer
Streaming output truncated to the last 5000 lines.
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029326.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029327.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029328.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029329.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029330.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029331.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029332.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029333.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029334.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029335.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029336.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029337.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029338.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029339.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029340.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029341.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029342.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029343.jpg  
  inflating: skin_cancer/ham10000_images_part_2/ISIC_0029344.jpg 
  
!rm -r /content/skin_cancer/ham10000_images_part_1
!rm -r /content/skin_cancer/ham10000_images_part_2
```
# Step 2 : Importing Essential Libraries
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
```
# Step 3 : Making Dictionary of images and labels
In this step I have made the image path dictionary by joining the folder path from base directory base_skin_dir and merge the images in jpg format from both the folders HAM10000_images_part1.zip and HAM10000_images_part2.zip
```python
dataset_dir = os.path.join('..', '/content/skin_cancer')

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(dataset_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
```

# Step 3 : Reading & Processing data
In this step we have read the csv by joining the path of image folder which is the base folder where all the images are placed named base_skin_dir. After that we made some new columns which is easily understood for later reference such as we have made column path which contains the image_id, cell_type which contains the short name of lesion type and at last we have made the categorical column cell_type_idx in which we have categorize the lesion type in to codes from 0 to 6

```python
skin_df = pd.read_csv(os.path.join(dataset_dir, 'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()
```
# Step 4 : Reading & Processing data
In this step we have read the csv by joining the path of image folder which is the base folder where all the images are placed named base_skin_dir. After that we made some new columns which is easily understood for later reference such as we have made column path which contains the image_id, cell_type which contains the short name of lesion type and at last we have made the categorical column cell_type_idx in which we have categorize the lesion type in to codes from 0 to 6

skin_df = pd.read_csv(os.path.join(dataset_dir, 'HAM10000_metadata.csv'))

- Creating New Columns for better readability
```python
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Now lets see the sample of tile_df to look on newly made columns
skin_df.head()

lesion_id	image_id	dx	dx_type	age	sex	localization	path	cell_type	cell_type_idx
0	HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
1	HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
2	HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
3	HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp	/content/skin_cancer/HAM10000_images_part_1/IS...	Benign keratosis-like lesions	2
4	HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear	/content/skin_cancer/HAM10000_images_part_2/IS...	Benign keratosis-like lesions	2
```

# Step 5 : Data Cleaning
In this step we check for Missing values and datatype of each field
```python
skin_df.isnull().sum()
lesion_id         0
image_id          0
dx                0
dx_type           0
age              57
sex               0
localization      0
path              0
cell_type         0
cell_type_idx     0
dtype: int64

skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
skin_df.isnull().sum()
lesion_id        0
image_id         0
dx               0
dx_type          0
age              0
sex              0
localization     0
path             0
cell_type        0
cell_type_idx    0
dtype: int64

skin_df.dtypes
lesion_id         object
image_id          object
dx                object
dx_type           object
age              float64
sex               object
localization      object
path              object
cell_type         object
cell_type_idx       int8
dtype: object
```

# Step 6 : EDA
In this we will explore different features of the dataset , their distrubtions and actual counts

Plot to see distribution of 7 different classes of cell type
```python
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
```
![cell type plot](https://user-images.githubusercontent.com/72849717/96358211-dbe02600-112e-11eb-862a-c1e195939769.png)

Its seems from the above plot that in this dataset cell type Melanecytic nevi has very large number of instances in comparison to other cell types

Plotting of Technical Validation field (ground truth) which is dx_type to see the distribution of its 4 categories which are listed below :
1. Histopathology(Histo): Histopathologic diagnoses of excised lesions have been performed by specialized dermatopathologists.
2. Confocal: Reflectance confocal microscopy is an in-vivo imaging technique with a resolution at near-cellular level , and some facial benign with a grey-world assumption of all training-set images in Lab-color space before and after manual histogram changes.
3. Follow-up: If nevi monitored by digital dermatoscopy did not show any changes during 3 follow-up visits or 1.5 years biologists accepted this as evidence of biologic benignity. Only nevi, but no other benign diagnoses were labeled with this type of ground-truth because dermatologists usually do not monitor dermatofibromas, seborrheic keratoses, or vascular lesions.
4. Consensus: For typical benign cases without histopathology or followup biologists provide an expert-consensus rating of authors PT and HK. They applied the consensus label only if both authors independently gave the same unequivocal benign diagnosis. Lesions with this type of groundtruth were usually photographed for educational reasons and did not need further follow-up or biopsy for confirmation.

```python
skin_df['dx_type'].value_counts().plot(kind='bar')
```
![dx_type_plot](https://user-images.githubusercontent.com/72849717/96358227-0f22b500-112f-11eb-9901-1026b4df151c.png)

