import json
import os
import glob
from skimage import io
from tqdm import tqdm
def read_label(label_txt):
    box_arr=[]
    with open(label_txt, "r") as f:
        for line in f.readlines():
            if len(line)==0:
                break
            line = line.strip('\n').split(' ')
            box_arr.append([eval(data) for data in line][1:])
    return box_arr
coco_instance_json_dict=dict(
        images=[],
        annotations=[],
        categories=[{'id':0, 'name': 'PW'}]
    )
train_filepath=r'../user_data/train_image/'
label_path=r'../user_data/train_label'
image_list=glob.glob(os.path.join(train_filepath,'*.tif'))
img_id=0
box_id=0
for image_path in tqdm(image_list):
    img=io.imread(image_path)
    file_name=os.path.split(image_path)[-1]
    h,w,c=img.shape
    assert (h,w)==(512,512)
    coco_instance_json_dict['images'].append({
        'id':img_id,'file_name':file_name,"height":h,"width":w
    })
    label=read_label(label_path+file_name.replace('.tif', '.txt'))
    for box in label:
        box_instance=[512*xy for xy in box]
        coco_instance_json_dict['annotations'].append({
            'image_id':img_id,'id':box_id,'category_id':0,'bbox':box_instance,'area':box_instance[2]*box_instance[3],
            'segmentation':[],'iscrowd':0
        })
        box_id=box_id+1
    img_id=img_id+1
out_json_path=r'../user_data/train_coco_pw.json'
with open(out_json_path, 'w') as output_json_file:
    json.dump(coco_instance_json_dict, output_json_file, indent=4)