# Copyright (c) Meta Platforms, Inc. and affiliates.

source .venv/bin/activate

# 1. Bridge matching pretraining
python -u -m wt_asbs.experiment.pretrain_bm \
    root=ckpts \
    experiment=ala2 \
    name=ala2_pretrain

# 2. WT-ASBS training
python -u -m wt_asbs.experiment.train_asbs \
    root=ckpts \
    experiment=ala2 \
    name=ala2_wt_asbs \
    pretrained_controller_checkpoint=ckpts/ala2_pretrain/checkpoints/epoch_1000.pt

# 3. Sampling from the final checkpoint
python -u -m wt_asbs.experiment.inference \
    checkpoint_directory=ckpts/ala2_wt_asbs \
    num_samples=1000000
