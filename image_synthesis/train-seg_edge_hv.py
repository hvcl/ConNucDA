import pytorch_lightning as pl
import argparse 
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='lizard')
parser.add_argument('--setting', type=str, default='base')
parser.add_argument('--resume_path', type=str, default='no')
args = parser.parse_args()

if args.dataset == 'lizard':
    from seg_edge_hv.lizard_dataset import MyDataset
elif args.dataset == 'pannuke':
    from seg_edge_hv.pannuke_dataset import MyDataset
elif args.dataset == 'endonuke':
    from seg_edge_hv.endonuke_dataset import MyDataset

# if args.resume_path is not 'no':
#     args.resume_path = ''

# endonuke 100k
# pannuke 100k
# lizard 100k

# python train-seg_edge_hv.py --dataset lizard --setting iter100k

# Configs
resume_path = f'./pathldm/control/control_plip_imagenet_ini_{args.dataset}_seg_edge_hv.ckpt'
batch_size = 16
logger_freq = 600
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

if args.setting=="iter50k_b32":
    max_steps = 50000
    batch_size = 32

elif args.setting=="iter20k":
    max_steps = 20000
    batch_size = 80
elif args.setting=="iter25k":
    max_steps = 25000
    batch_size = 64
elif args.setting=="iter50k":
    max_steps = 50000
elif args.setting=="iter75k":
    max_steps = 75000
elif args.setting=="iter100k":
    max_steps = 100000
elif args.setting=="iter150k":
    max_steps = 150000
else:
    max_steps = 200000



# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(f'./models/cond_seg_edge_hv/cldm_pathLDM_{args.dataset}.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_steps=max_steps, weights_save_path=f"./lightning_logs/{args.dataset}-setting_{args.setting}")


# Train!
trainer.fit(model, dataloader)

