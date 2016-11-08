
python alexnet_train.py --train_dir=model_1_relu --activation=relu --learning_rate=0.0001 --steps_per_checkpoint=1000 --model=1 --data_dir=data/ --batch_size=256 --eval_dir=results/ --eval --num_examples=10000

python alexnet_train.py --train_dir=model_1_tanh --activation=tanh --learning_rate=0.0001 --steps_per_checkpoint=1000 --model=1 --data_dir=data/ --batch_size=256 --eval_dir=results/ --eval --num_examples=10000

python alexnet_train.py --train_dir=model_small --activation=relu --learning_rate=0.001 --steps_per_checkpoint=1000 --model=5 --data_dir=data/ --train --eval_dir=results/ --eval --num_examples=10000

