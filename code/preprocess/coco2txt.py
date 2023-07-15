import json
import numpy as np
import os
predict_coco=r'PW_log/swin_b_htc_pafpn_adawm_1x_auto_aug_gn_ws_2x/infer_epoch21_test_flip_softnms_rpn_rcnn.bbox.json'
test_coco=r'../user_data/fusai_test_false.json'
infer_path=r'../prediction_result/result'
if not os.path.exists(infer_path):
    os.makedirs(infer_path)
with open(predict_coco,'r') as load_f:
    load_dict_predict_coco = json.load(load_f)
with open(test_coco,'r') as load_f:
    load_dict_test_coco = json.load(load_f)
pred_box_id_index=[]
for box in load_dict_predict_coco:
    pred_box_id_index.append(box['image_id'])
pred_box_id_index=np.array(pred_box_id_index)
def save_str_list_txt(str_list,filepath):
    with open(filepath, 'w') as f:
        if len(str_list)==0:
            f.write('')
        else:
            for i in str_list:
                f.write(i + '\n')
for coco_img in load_dict_test_coco['images']:
    img_id=coco_img['id']
    img_name=coco_img['file_name']
    txt_name=img_name.replace('.tif','.txt')
    filepath_txt=os.path.join(infer_path,txt_name)
    tmp = np.where(pred_box_id_index == img_id)
    index_list = tmp[0].tolist()
    str_list=[]
    if len(index_list)>0:
        for index in index_list:
            box=load_dict_predict_coco[index]
            box_infer=box['bbox']
            conf=box['score']
            box_norm=[xy/512 for xy in box_infer]
            line_str=str(box['category_id'])+' '+str(box_norm[0])+' '+str(box_norm[1])+' '+str(box_norm[2])+' '+str(box_norm[3])+' '+str(conf)
            str_list.append(line_str)
    save_str_list_txt(str_list, filepath_txt)
