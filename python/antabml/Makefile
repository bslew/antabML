train:
	python ./train_antab.py --bs 100 --epoch 100 --model dense --model_dir ../../models/dense-dsize_5000-split_0.8_0.2_0/ --train_dir ../../data/train/ --load_worker 1 --dsize 5000 --denseConf 2500 500 500 2500 --split 0.8 0.2 0

test:
	python ./test_antab.py  --model_file ../../models/dense-dsize_5000-split_0.8_0.2_0/model.ckp_100 --test_file ../../data/test/blew-Nov21-ea065btr.01.awpkl

