
vocabs:
	python -m utils.vocab --data_dir sents.txt --vocab_path vocab-sents.pkl --cutoff 30000
	python -m utils.vocab --data_dir labels.txt --vocab_path vocab-labels.pkl --cutoff 30000
	python -m utils.vocab --data_dir sents_val.txt --vocab_path vocab-sents-val.pkl --cutoff 30000
	python -m utils.vocab --data_dir labels_val.txt --vocab_path vocab-labels-val.pkl --cutoff 30000

train:
	python train.py \
		--feats_path sents.txt \
		--feats_vocab vocab-sents.pkl \
		--labels_path labels.txt \
		--labels_vocab vocab-labels.pkl \
		--hidden_dim=128 \
		--save_dir=foo \
		--word_dim=256 \
		--wordembed_type=none \
		--wordembed_path=./ \
		--wordembed_freeze=False \
		--config=conf.yml \
		--batch_size=32  \
		--save \
		--save_period 10 \
		--n_epochs 30 \
		--gpu

train_with_fasttext:
	python train.py \
		--feats_path sents.txt \
		--feats_vocab vocab-sents-w-unlabelled.pkl \
		--labels_path labels.txt \
		--labels_vocab vocab-labels-w-unlabelled.pkl \
		--hidden_dim=128 \
		--save_dir=foo \
		--word_dim=300 \
		--wordembed_type=fasttext \
		--fasttext_path=$(HOME)/Code/fastText/fasttext \
		--wordembed_path=./embeddings/wiki.de.bin \
		--wordembed_freeze=False \
		--config=conf.yml \
		--batch_size=32 \
		--save \
		--save_period 10 \
		--n_epochs 30 \
		--gpu \
		--val \
		--val_period 10 \
		--val_feats_path sents_val_w_unlabelled.txt \
		--val_labels_path labels_val_w_unlabelled.txt \
		--text_preview

predict:
	python predict.py \
		--ckpt_path foo/20170922-172008-main/model-37833-0.6629 \
		--feats_path sents.txt \
		--feats_vocab vocab-sents.pkl \
		--labels_vocab vocab-labels.pkl \
		--word_dim=256 \
		--hidden_dim=128 \
		--save_dir ./output

start_visdom:
	python -m visdom.server
