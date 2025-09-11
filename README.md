## PaVeRL-SQL

**PaVeRL-SQL** is the repository for paper "PaVeRL-SQL: Text–to–SQL via Partial–Match
Rewards and Verbal Reinforcement Learning" (https://arxiv.org/abs/2509.07159)

#### Environment
Generally the pipeline works with pytorch>=2.6.0, transformers>=4.48.2, and vllm>=0.8.2, if newer backbone model is needed, the corresponding packages need to satisfy the model requirement as well. For verl installation, we follow the verl documents as following:
```bash
git clone https://github.com/volcengine/verl && cd verl
pip install -e .
```

To customize the learning rate scheduler and two stage training, we modified the following code from the original verl package code:
```bash
./verl/verl/workers/fsdp_workers.py
./verl/verl/workers/actor/dp_actor.py
./verl/verl/utils/torch_functional.py
./verl/verl/utils/logger/aggregate_logger.py
./verl/verl/trainer/main_ppo.py
./verl/verl/trainer/config/ppo_trainer.yaml
```
#### File Structure
The verbal-RL pipeline codes are at "./verbal_rl", the CoT RL training code are at "./", we give the SynSQL-10K sample as an example. After GRPO training, use the code "modelhf.py" to consolidate the model. The inference and evaluation code are at "./infer_eval", the official evaluation code are modified based on OmniSQL github (https://github.com/RUCKBReasoning/OmniSQL). 
