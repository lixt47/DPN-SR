nohup python main.py --datapath ./data/14lap/ --pretrain True > train_pre_14lap.log 2>&1 &
nohup python main.py --lr 0.000005 --datapath ./data/14lap/ --start checkpoints/{experiment_id}/model > train_rl_14lap.log 2>&1 &
nohup python main.py --datapath ./data/14res/ --pretrain True > train_pre_14res.log 2>&1 &
nohup python main.py --lr 0.000005 --datapath ./data/14res/ --start checkpoints/{experiment_id}/model > train_rl_14res.log 2>&1 &
nohup python main.py --datapath ./data/15res/ --pretrain True > train_pre_15res.log 2>&1 &
nohup python main.py --lr 0.000005 --datapath ./data/15res/ --start checkpoints/{experiment_id}/model > train_rl_15res.log 2>&1 &
nohup python main.py --datapath ./data/16res/ --pretrain True > train_pre_16res.log 2>&1 &
nohup python main.py --lr 0.000005 --datapath ./data/16res/ --start checkpoints/{experiment_id}/model > train_rl_16res.log 2>&1 &


