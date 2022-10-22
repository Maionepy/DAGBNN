## Overview
This is the implementation of the model-based RL algorithm related to the semester project "Data-Efficient Model Baseed Reinforcement Learning for Robotics and Control" (Bruno F. Maione with supervision of M.Sc. Cong Li), based in the works of the method MBPO in pytorch as described in the following paper: [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) and in the implementation of [MBPO - Pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch)

## Dependencies

MuJoCo 2.10

## Usage

#### MBPO

> python main_mbpo.py --env_name 'Walker2d-v2' --num_epoch 300 --model_type 'pytorch'

> python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 300 --model_type 'tensorflow'

> python main_mbpo.py --env_name 'InvertedPendulum-v2' --num_epoch 15 --model_type 'tensorflow'

#### DAGBNN

> python main_dagbnn.py --env_name 'InvertedPendulum-v2' --num_epoch 15 --model_type 'tensorflow' --rollout_fixed

#### SAC

> python main_sac.py --env_name 'InvertedPendulum-v2'

## Reference
* Official tensorflow implementation: https://github.com/JannerM/mbpo
* Code to the reproducibility challenge paper: https://github.com/jxu43/replication-mbpo
* Code closing the gap of Pytorch implementation: https://github.com/Xingyu-Lin/mbpo_pytorch