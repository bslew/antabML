
model=dense
epoch=500
bs=1000
dsize=2500
denseConf=1500 500 500 1500
dropout=0.1
loss=smoothL1
lr=0.001
split=0.5 0.5 0
seed=0
train_dir=~/projects/antab-vlbeer-data/wisdom
model_dir=../../models/auto
load_workers=4
chkpt_save=500
MLflow_tracking_uri=http://192.168.1.63:5000
MLflow_run_name=dense 2500 2 DS=all

train:
#	python ./train_antab.py --bs 100 --epoch 100 --model dense --model_dir ../../models/dense-dsize_5000-split_0.8_0.2_0/ --train_dir ../../data/train/ --load_worker 1 --dsize 5000 --denseConf 2500 500 500 2500 --split 0.8 0.2 0
	
	. ../../venv/bin/activate && python ./train_antab.py --bs ${bs} --epoch ${epoch} --model ${model} \
	--dsize ${dsize} \
	--denseConf ${denseConf} \
	--dropout ${dropout} \
	--loss ${loss} \
	--lr ${lr} \
	--split ${split} \
	--split_seed ${seed} \
	--train_dir ${train_dir} \
	--model_dir ${model_dir} \
	--load_workers ${load_workers} \
	--chkpt_save ${chkpt_save} \
	--MLflow_tracking_uri "${MLflow_tracking_uri}" \
	--MLflow_run_name "${MLflow_run_name}"

test:
	python ./test_antab.py  --model_file ../../models/dense-dsize_5000-split_0.8_0.2_0/model.ckp_100 --test_file ../../data/test/blew-Nov21-ea065btr.01.awpkl

