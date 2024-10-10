import math
from shutil import copyfile

from tqdm import tqdm
import pandas as pd
import os

# df_isic=pd.read_csv(r'/kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv')
# print (df_isic.columns)
# # Add .jpg extension to the image filenames
# df_isic['image']=df_isic['image'].apply(lambda x: x+ '.jpg')
# x = df_isic.head()
# print (df_isic.columns[1:])
#
# # Iterate over dataframe and move images to correct folders
# for index, row in tqdm(df_isic.iterrows(), total=df_isic.shape[0], desc=f'Copying ISIC 2019 dataset images..'):
#     # Get the image pathname
#     hot_label = row[row == 1].index.tolist()[0]
#     image_name = row['image']
#     a = ['MEL', 'BCC', 'AKIEC', 'VASC','AK','SCC']
#     if hot_label == 'NV' or hot_label == 'UNK':
#         pass
#     elif hot_label in a:
#         hot_label = 'malignant'
#         src_path = os.path.join("/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input", image_name)
#         dst_path = os.path.join(data_dir, hot_label, image_name)
#         copyfile(src_path, dst_path)
#     else:
#         hot_label = 'benign'
#         src_path = os.path.join("/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input", image_name)
#         dst_path = os.path.join(data_dir, hot_label, image_name)
#         copyfile(src_path, dst_path)
#
#
# tot = 0
# for label in label_names:
#     cnt_label = len(os.listdir(os.path.join(data_dir, label)))
#     print(f"There are {cnt_label} images with label {label}.")
#     tot += cnt_label
# print(f"\nThere are {tot} total images across all labels.")

input_dir = "/Users/andrey/kaggle/input/archive"
output_dir = "/Users/andrey/kaggle/input/dataset"

labels = [f for f in os.listdir(input_dir)]

malignant = ['MEL', 'BCC', 'AKIEC', 'VASC', 'AK', 'SCC']

p = 0.2
# split_size = 5000

os.makedirs(f"{output_dir}/train/malignant")
os.makedirs(f"{output_dir}/test/benign")
os.makedirs(f"{output_dir}/train/benign")
os.makedirs(f"{output_dir}/test/malignant")

for label in labels:
    files = [f for f in os.listdir(f"{input_dir}/{label}")]
    test_size = math.ceil(len(files) * p)

    train_files = files[:-test_size]
    test_files = files[-test_size:]

    for train_file in train_files:
        src = f"{input_dir}/{label}/{train_file}"

        if label in malignant:
            dst = f"{output_dir}/train/malignant/{train_file}"
        else:
            dst = f"{output_dir}/train/benign/{train_file}"

        copyfile(src, dst)

    for test_file in test_files:
        src = f"{input_dir}/{label}/{test_file}"

        if label in malignant:
            dst = f"{output_dir}/test/malignant/{test_file}"
        else:
            dst = f"{output_dir}/test/benign/{test_file}"

        copyfile(src, dst)
