import numpy as np
import cv2
import pickle
import string


def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 7) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': float(x1), 'x2': float(x2), 'y1': float(y1), 'y2': float(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


# Code to go through tsv files and find only the ones needed to download
path_to_directory = "C:\\Users\\Mukesh\\GoogleAI\\DataFiles_OD\\"

train_images = {}
with open(path_to_directory + "train_images_boxable_with_rotation.csv", "r", encoding="utf8") as f:
    for l in f:
        r = l.split(",")
        if r[0] == 'ImageID':
            continue
        train_images[r[2]] = r[0]

print("loaded {} image ids".format(len(train_images)))

# screen the images to include only the ones in the training set
path = "C:\\Users\\Mukesh\\GoogleAI\\DataFiles\\"
infn = path + "open-images-dataset-train9.tsv"
count = 0
outf = open(infn + "trimmed_", "w+")

with open(infn, "r") as f:
    for l in f:
        r = l[:-1].split("\t")
        if len(r) == 1:  # header
            outf.write(l)
            continue
        try:
            id = train_images[r[0]]
            outf.write(l)
            count += 1
        except:
            continue
#        print(r[2])
outf.close()
print(count)

# converting a csv to list
import pandas as pd

path = "C:\\Users\\Mukesh\\GoogleAI\\DataFiles_OD\\"
columns = ['mid', 'desc']
read = pd.read_csv(path + "class_descriptions_500.csv", names=columns)
read = pd.DataFrame(read)
read = read.drop(columns=['mid'])
read = read['desc'].tolist()

import csv

path2 = "C:\\Users\\Mukesh\\GoogleAI\\Model_6_ODmxnet\\mx_rcnn\\data\\OIDdevkit\\"

with open(path2 + "class names.txt", 'w') as f:
    wr = csv.writer(f, delimiter="\n")
    for ele in read:
        wr.writerow(["'{}'".format(ele) + ','])

with open(path2 + "download names.txt", 'w') as f:
    for item in downloaded_list:
        f.write("%s\n" % item)
