cd code
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    PW_log/swin_b_htc_pafpn_adawm_1x_auto_aug_gn_ws_2x/htc_swin_b_pafpn_auto_aug_1x_pretrained_gn_ws_2x.py \
    PW_log/swin_b_htc_pafpn_adawm_1x_auto_aug_gn_ws_2x/best_bbox_mAP_epoch_21.pth \
    --format-only \
    --options "jsonfile_prefix=PW_log/swin_b_htc_pafpn_adawm_1x_auto_aug_gn_ws_2x/infer_epoch21_test_flip_softnms_rpn_rcnn"
python preprocess/coco2txt.py