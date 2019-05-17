import pandas as pd
import json


# read csv and assign headers
headers1 = ['LabelName', 'ClassDescription']
class_description_boxable = pd.read_csv(path + "class_descriptions_boxable.csv", names = headers1)

# Preprocessing Data for OD
# make a list with the downloaded images
import pandas as pd
import os
import csv

# goes to the directory of the train images and creates a list of the downloaded images
directory = "C:\\Users\\Mukesh\\GoogleAI\\DataFiles\\challenge2018_train\\"
downloaded_list = []
for filename in os.listdir(directory):
    downloaded_list.append(filename)

# strips the imgs of the .jpg tag, see if you can add it to the for loop above
downloaded_list = [imgs.strip('.jpg') for imgs in downloaded_list]

# convert the downloaded imgs list to a dataframe and load the annotations df
path_to_datafiles = "C://Users//Mukesh//GoogleAI//DataFiles_VRD//"
annotations = pd.read_csv(path_to_datafiles + "Annotations_boxes.csv")
annotations = pd.DataFrame(annotations)

# create a new df with descriptions from annotations for the downloaded images
downloaded_df_train = annotations[annotations['ImageID'].isin(downloaded_list)]

# drop columns not needed for the model
drop_columns = ['Source', 'LabelName', 'Confidence', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction',
                'IsInside']
downloaded_df_train = downloaded_df_train.drop(columns=drop_columns)

# create a dictionary with MID labels to label names
descriptions_file = csv.reader(open(path_to_datafiles + "class_descriptions_boxable.csv", 'r'))
descriptions_file_dict = {}
for row in descriptions_file:
    k, v = row
    descriptions_file_dict[k] = v

# convert the MID LabelNames to human readable Label names
downloaded_df_train['Label'] = downloaded_df_train.Label.map(descriptions_file_dict)

# adding file path as a prefix and .jpg as a suffix to the ImageID to fit the input file
downloaded_df_train['ImageID'] = downloaded_df_train['ImageID'].apply(lambda x: "{}{}{}".format(directory, x, '.jpg'))

# reorder columns to fit the input file
downloaded_df_train_clean = downloaded_df_train[['ImageID', 'XMin', 'YMin', 'XMax', 'YMax', 'Label']]

# export the df as a txt file
downloaded_df_train_clean.to_csv('challenge2018_train.txt', index=False, header=False)

# make a column appending Label1Name + Relationship + Label2Name

rel_triplet_annotations2['Img_desc'] = rel_triplet_annotations2['LabelName1'] + ' ' + \
                                       rel_triplet_annotations2['RelationshipLabel'] + ' ' + \
                                       rel_triplet_annotations2['LabelName2']

# multiply a column by a specific value
train['XMin1'] = train['XMin1'].apply(lambda x: int(x*width))

# subtract a two columns to create a new one
train ['box1length'] = train['XMax1'] - train['XMin1']

# unique labels in a column
train['RelationshipLabel'].unique()

# rename columns
COLUMN_NAMES = {"RelationshipLabel_at": "at",
                "RelationshipLabel_hits": "hits"}

train = train.rename(columns=COLUMN_NAMES)

# To see number of unique values in a column
metadata_imageid_train_val_set['ImageID'].nunique()

# parse a json file and replace class_id labels with mid labels
from pandas.io.json import json_normalize

id_to_midlabels = {}

csv_file = 'challenge-2018-class-descriptions-500.csv'
meta_dir = os.path.join(main_dir, challenge2018)
boxable_classes_descriptions = os.path.join(meta_dir, csv_file)

i = 0
with open(boxable_classes_descriptions, 'r') as descriptions_file:
	for row in csv.reader(descriptions_file):
		if len(row):
			label = row[0]
			# description = row[1].replace("\"", "").replace("'", "").replace('`', '')
			id_to_midlabels[i] = label
			i += 1

base_dir = os.path.join(main_dir, images, test)

with open("predict_file.json", 'r') as f:
	d = json.load(f)
	with open("submission_file.json", 'w') as sub_file:
		for img in os.listdir(base_dir):
			img = img.strip('.jpg')
			train = json_normalize(d[img], record_path='boxes')
			train['cls_id'] = train.cls_id.map(id_to_midlabels)
			json.dump(train, sub_file)
			

# to split a dataframe
test = train[:99999]

# drop column
df.drop(columns=['B', 'C'])

# rename column
df = df.rename(columns={"A": "a", "B": "b"})

# to rename csv without header
header = ["PredictionString"]
pqr = pd.read_csv(q, names=header)

# drop specific rows based on a value
vr_iou_negative = vr[vr['iou'] < 0 ]
vr = vr.drop(vr_iou_negative.index, axis=0)

# to check for empty columns
np.where(pd.isnull(df))

# convert a list to a df
vr_df = pd.DataFrame({'ImageId':downloaded_list})

# merge two df with using left to join with nan values 
result4 = pd.merge(vr_df, vr, on='ImageId', how = 'left')

# access row by column value
od_mock.loc[od_mock['ImageId'] == '8d65ca08cb5ce8e8']

# drop specific columns with nan values
anns = anns.dropna(subset=['name', 'born'])

# group the rows based on column id
customer_id_count = orders_2017.groupby(['customer_id', 'gender']).agg(['nunique']).reset_index()
customer_id_count['order_count'] = customer_id_count['value']
customer_id_count = customer_id_count._drop_axis(labels=['value', 'date'], axis=1)

# reset index
comments = comments.reset_index(drop=True)
