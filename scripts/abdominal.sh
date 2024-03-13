domains=('CHAOST2')

for dm in "${domains[@]}"
do
   python train.py \
   --expname     'dcon' \
   --phase       'train' \
   --gpu_ids      '1' \
   --f_seed       1 \
   --f_determin   1 \
   --lr           0.0005 \
   --model        'unet' \
   --batchSize    20 \
   --all_epoch    1200 \
   --validation_freq 50 \
   --testfreq     50 \
   --display_freq 2000 \
   --data_name    'ABDOMINAL' \
   --nclass       5 \
   --tr_domain    $dm \
   --consist_f    1 \
   --contrast_f   1 \
   --fmethod      'asymr' \
   --save_prediction True \
   --w_ce         1.0 \
   --w_dice       1.0 \
   --w_seg        1.0 \
   --w_consist    1.0 \
   --w_contrast   1.0 \
   --num_augs1    6 \
   --augflag1     False \
   --f_dropout1   0 \
   --dropout_rate1 0.0 \
   --f_dropout2   1 \
   --dropout_rate2 0.5 \
   --gls_nlayer    4 \
   --gls_interm    2 \
   --gls_outnorm   'frob' \
   --glsmix_f      1 \
   --mixalpha      0.2 \
   --temperature   0.05 \
   --n_view        10 
done
