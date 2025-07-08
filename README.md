# Learning Vision-based Policy for High-Speed Autonomous Racing

This is the repository of the F1TENTH Gym environment, aiming to train vision-based racing policy.

This project is still under heavy developement.

## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
conda create -n visual_racing python=3.8
conda activate visual_racing
git clone https://github.com/optilearn472/visual_racing.git
cd visual_racing
pip install -e .
```

You can run a teacher policy training example by:
```bash
cd examples/rl_race
python3 train_teacher.py
```

You can run a student policy training example by:
```bash
cd examples/rl_race
python3 train_dagger_with_dagger.py
```

## Citing
This code is based on the F1TENTH Gym environment. https://github.com/f1tenth/f1tenth_gym.git