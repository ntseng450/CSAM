python train.py --dataroot ./datasets/summer2winter_yosemite --name s2wy_CSAM_STD3 --CUT_mode CUT --gpu_ids 0 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --save_epoch_freq 100
exit
python train.py --dataroot ./datasets/summer2winter_yosemite --name s2wy_CSAM_STD3 --CUT_mode CUT --gpu_ids 0 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --save_epoch_freq 100
ls
exit
python train.py --dataroot ./datasets/summer2winter_yosemite --name s2wy_CSAM_STD3 --CUT_mode CUT --gpu_ids 0 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --save_epoch_freq 100
nohup python train.py --dataroot ./datasets/summer2winter_yosemite --name s2wy_CSAM_STD3 --CUT_mode CUT --gpu_ids 0 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --save_epoch_freq 100
python train_attention.py --dataroot /app/o2a_CSAM --name o2a_CSAM --CUT_mode CUT --display_id 0 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 200 --gpu_ids 0
ls
exit
python train_attention.py --dataroot /app/o2a_CSAM --name o2a_CSAM --CUT_mode CUT --display_id 0 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 200 --gpu_ids 0
ls
cd ..
ls
cd app
cd ..
ls
cd CSAM/
ls
cd app
exit
python train_attention.py --dataroot /app/o2a_CSAM --name o2a_CSAM --CUT_mode CUT --display_id 0 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 200 --gpu_ids 0
nohup python train_attention.py --dataroot /app/o2a_CSAM --name o2a_CSAM --CUT_mode CUT --display_id 0 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 200 --gpu_ids 0
nohup python train.py --dataroot /app/s2wy_CSAM --name s2wy_CSAM_800 --CUT_mode CUT --display_id 0 --gpu_ids 3 --n_epochs 400 --n_epochs_decay 400 --save_epoch_freq 800
exit
python generate.py --CUT_mode CUT --name s2wy_CSAM_800 --dataroot datasets/s2wy_test/ --phase test
ls
exit
exit
