import argparse
import os
import time

import torch
import torchvision
import torchvision.transforms as T
import wandb
from PIL import Image

from datasets.consep import ConsepDataset, ToTensorNoNorm
from datasets.consep import mapping_id as mapping_id_consep
from datasets.consep import transform_lbl as transform_lbl_consep

from datasets.lizard import LizardDataset #, ToTensorNoNorm
from datasets.lizard import mapping_id as mapping_id_lizard
from datasets.lizard import transform_lbl as transform_lbl_lizard

from datasets.pannuke import PannukeDataset #, ToTensorNoNorm
from datasets.pannuke import mapping_id as mapping_id_pannuke
from datasets.pannuke import transform_lbl as transform_lbl_pannuke

from datasets.endonuke import EndonukeDataset #, ToTensorNoNorm
from datasets.endonuke import mapping_id as mapping_id_endonuke
from datasets.endonuke import transform_lbl as transform_lbl_endonuke

from imagen_pytorch import BaseJointUnet, JointImagen, JointImagenTrainer, SRJointUnet
from imagen_pytorch.imagen_pytorch import NullUnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_iters', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=300000)
    parser.add_argument('--log_every', type=int, default=10000)
    parser.add_argument('--save_every', type=int, default=10000) #10000
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)

    parser.add_argument('--lr', type=float, default=1.2e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=16)
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--lambdas', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--pred_objectives', type=str, default='noise')

    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--project', type=str, default='imagen')
    parser.add_argument('--exp_name', type=str, default='imagen')
    parser.add_argument('--model_type', type=str, default='base_256x256')
    parser.add_argument('--diffusion_type', type=str, default='joint')
    parser.add_argument('--random_crop_size', type=int, nargs='*', default=(256, 512))
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--no_condition_on_text', action='store_false', dest='condition_on_text')
    # parser.set_defaults(condition_on_text=True)

    parser.add_argument('--lowres_max_thres', type=float, default=0.999, help='lowres augmentation maximum')
    parser.add_argument('--augmentation_type', type=str, default='flip')
    parser.add_argument('--noise_schedules', type=str, nargs='*', default=('cosine', ))
    parser.add_argument('--noise_schedules_lbl', type=str, nargs='*', default=('cosine_p', ))
    parser.add_argument('--cosine_p_lbl', type=float, default=1.0)

    parser.add_argument('--channels_lbl', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=20)
    # cityscapes: 20, celeba: 19 (include background)

    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--split', type=str, default='100')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--caption_list_dir', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--test_caption_files', type=str, nargs='*',
                        default='data/eval_samples/cityscapes/frankfurt_000000_000294.txt')
    parser.add_argument('--start_image_or_video', type=str, nargs='*',
                        default=['data/eval_samples/cityscapes/frankfurt_000000_000294_leftImg8bit.png', ])
    parser.add_argument('--start_label_or_video', type=str, nargs='*',
                        default=['data/eval_samples/cityscapes/frankfurt_000000_000294_gtFine_labelIds.png', ])
    parser.add_argument('--test_batch_size', type=int, default=1)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    args.side_x, args.side_y = [int(i) for i in args.model_type.split('_')[-1].split('x')]
    if args.dataset not in ["consep", "lizard", "pannuke", "endonuke"]:
        if len(args.test_caption_files) == 1 and args.test_batch_size != 1:
            args.test_caption_files = args.test_caption_files * args.test_batch_size
        args.test_caption = []
        for test_caption_file in args.test_caption_files:
            with open(test_caption_file, 'r') as f:
                args.test_caption.append(f.read())
        assert len(args.test_caption) == args.test_batch_size

    assert args.test_batch_size <= args.max_batch_size
    if len(args.random_crop_size) == 0:
        args.random_crop_size = None
    else:
        assert len(args.random_crop_size) == 2, args.random_crop_size
        args.random_crop_size = tuple(args.random_crop_size)

    # directories
    if args.resume:
        args.checkpoint_dir = os.path.dirname(args.resume)
        args.start_iters = int(os.path.basename(args.resume).split('.')[1])
        args.exp_name = '_'.join(os.path.basename(args.checkpoint_dir).split('_')[2:])
    else:
        strtime = time.strftime("%y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.diffusion_type,
                                           args.model_type, f'{strtime}_{args.exp_name}')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    ####### FOR EXP
    args.condition_on_text = True
    if args.exp_name=="ratio-class_256x256":
        if args.dataset=='lizard':
            args.test_caption = [
                "pathology colon tissue label; nucleus proportion 0.02; including lymphocyte, plasma, neutrophil, and connective tissue", 
                "pathology colon tissue label; nucleus proportion 0.15; including neutrophil, epithelial, lymphocyte, plasma, and connective tissue", 
                "pathology colon tissue label; nucleus proportion 0.35; including neutrophil, epithelial, and connective tissue",
                "pathology colon tissue label; nucleus proportion 0.45; including neutrophil, epithelial, lymphocyte, plasma, and connective tissue"
            ]
            args.sample_caption = "0.02 (ORCP) 0.15 (RGOPC) 0.35 (RGC) 0.45 (RGOPC)"
        elif args.dataset=='pannuke':
            args.test_caption = [
                "pathology breast tissue label; nucleus proportion 0.02; including neoplastic and epithelial",
                "pathology esophagus tissue label; nucleus proportion 0.12; including neoplastic, inflammatory, and connective tissue", 
                "pathology colon tissue label; nucleus proportion 0.28; including neoplastic, inflammatory, and dead",
                "pathology headneck tissue label; nucleus proportion 0.45; including neoplastic and epithelial"
            ]
            args.sample_caption = "0.02 (PG) 0.12 (PCB) 0.28 (PCY) 0.45 (PG)"
        elif args.dataset=='endonuke':
            args.test_caption = [
                "pathology endometrium tissue label; nucleus proportion 0.05; including stroma and epithelium",
                "pathology endometrium tissue label; nucleus proportion 0.12; including stroma, epithelium, and others", 
                "pathology endometrium tissue label; nucleus proportion 0.28; including stroma and epithelium",
                "pathology endometrium tissue label; nucleus proportion 0.45; including stroma"
            ]
            args.sample_caption = "0.05 (GB) 0.12 (RGB) 0.28 (GB) 0.45 (B)"
    elif args.exp_name=="ratio-256x256":
        if args.dataset=='lizard':
            args.test_caption = [
                "pathology colon tissue label; nucleus proportion 0.02;", 
                "pathology colon tissue label; nucleus proportion 0.15;", 
                "pathology colon tissue label; nucleus proportion 0.35;",
                "pathology colon tissue label; nucleus proportion 0.45;"
            ]
            args.sample_caption = "0.02 (ORCP) 0.15 (RGOPC) 0.35 (RGC) 0.45 (RGOPC)"
        elif args.dataset=='pannuke':
            args.test_caption = [
                "pathology breast tissue label; nucleus proportion 0.02;",
                "pathology esophagus tissue label; nucleus proportion 0.12;", 
                "pathology colon tissue label; nucleus proportion 0.28;",
                "pathology headneck tissue label; nucleus proportion 0.45;"
            ]
            args.sample_caption = "0.02 (PG) 0.12 (PCB) 0.28 (PCY) 0.45 (PG)"
        elif args.dataset=='endonuke':
            args.test_caption = [
                "pathology endometrium tissue label; nucleus proportion 0.05;",
                "pathology endometrium tissue label; nucleus proportion 0.12;",
                "pathology endometrium tissue label; nucleus proportion 0.28;",
                "pathology endometrium tissue label; nucleus proportion 0.45;"
            ]
            args.sample_caption = "0.02 / 0.12 / 0.28 / 0.45"

    print(args)

    return args


def main():
    args = parse_args()

    # unet for imagen
    print('Creating JointUNets..')

    if args.model_type.startswith('base'):
        addi_kwargs = dict()
        addi_kwargs.update(dict(
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
            if args.condition_on_text else False,
        ))
        unet1 = BaseJointUnet(channels_lbl=args.channels_lbl, num_classes=args.num_classes, **addi_kwargs)
        unets = (unet1, )
        h1, w1 = [int(i) for i in args.model_type.split('_')[1].split('x')]
        image_sizes = ((h1, w1), )
        random_crop_sizes = (None, )
        args.unet_number = 1
    elif args.model_type.startswith('sr'):
        addi_kwargs = dict()
        if not args.condition_on_text:
            addi_kwargs.update(layer_cross_attns=False)
        unet1 = NullUnet()
        unet2 = SRJointUnet(channels_lbl=args.channels_lbl, num_classes=args.num_classes, **addi_kwargs)
        unets = (unet1, unet2)
        h1, w1 = [int(i) for i in args.model_type.split('_')[1].split('x')]
        h2, w2 = [int(i) for i in args.model_type.split('_')[2].split('x')]
        image_sizes = ((h1, w1), (h2, w2))
        # random_crop_sizes = (None, args.random_crop_size)
        random_crop_sizes = (None, None)
        args.unet_number = 2
    else:
        raise NotImplementedError(args.model_type)

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = JointImagen(
        unets=unets,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        random_crop_sizes=random_crop_sizes,
        num_classes=args.num_classes,
        timesteps=1000,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text=args.condition_on_text,
        lowres_max_thres=args.lowres_max_thres,
        pred_objectives=args.pred_objectives,
        noise_schedules=args.noise_schedules,
        noise_schedules_lbl=args.noise_schedules_lbl,
        cosine_p_lbl=args.cosine_p_lbl,
    )

    trainer = JointImagenTrainer(
        imagen,
        lr=args.lr,
        fp16=args.fp16,
        checkpoint_every=args.save_every,
        checkpoint_path=args.checkpoint_dir,
        max_checkpoints_keep=3,
        lambdas=args.lambdas,
    )
    args.world_size = trainer.accelerator.num_processes
    if args.resume:
        trainer.load(args.resume)
    print('Done!')

    print('Create Dataset...')
    if args.dataset == 'consep':
        dataset = ConsepDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            augmentation_type=args.augmentation_type,
        )
        mapping_id = mapping_id_consep
        transform_lbl = transform_lbl_consep
    
    elif args.dataset == 'lizard':
        dataset = LizardDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            caption_list_dir=args.caption_list_dir,
            augmentation_type=args.augmentation_type,
            exp_name=args.exp_name
        )
        mapping_id = mapping_id_lizard
        transform_lbl = transform_lbl_lizard
    
    elif args.dataset == 'pannuke':
        dataset = PannukeDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            caption_list_dir=args.caption_list_dir,
            augmentation_type=args.augmentation_type,
            exp_name=args.exp_name            
        )
        mapping_id = mapping_id_pannuke
        transform_lbl = transform_lbl_pannuke

    elif args.dataset == 'endonuke':
        dataset = EndonukeDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            caption_list_dir=args.caption_list_dir,
            augmentation_type=args.augmentation_type,
            exp_name=args.exp_name            
        )
        mapping_id = mapping_id_endonuke
        transform_lbl = transform_lbl_endonuke        
    
    else:
        raise NotImplementedError(args.dataset)
    
    trainer.add_train_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    print('Done!')

    # for up- / upup- sampler they need an image and a label for upsample
    start_image_or_video = start_label_or_video = None
    if args.unet_number > 1:
        start_image_or_video, start_label_or_video = [], []
        for img_path, lbl_path in zip(args.start_image_or_video, args.start_label_or_video):
            start_image_or_video.append(T.Compose([
                T.Resize(image_sizes[0], T.InterpolationMode.NEAREST),
                T.ToTensor()]
            )(Image.open(img_path)))
            start_label_or_video.append(
                mapping_id[T.Compose([
                    T.Resize(image_sizes[0], T.InterpolationMode.NEAREST),
                    ToTensorNoNorm()]
                )(Image.open(lbl_path)).long()].float()
            )
        start_image_or_video = torch.stack(start_image_or_video)
        start_label_or_video = torch.stack(start_label_or_video)

    if trainer.is_main and args.log_wandb:
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.exp_name,
            config=args,
            id=os.path.basename(args.checkpoint_dir),
            dir='wandb_dir',
        )

    print('Start Training...')
    # feed images into imagen, training each unet in the cascade
    for i in range(args.start_iters, args.num_iters):
        if i % args.print_every == 0 and trainer.is_main:
            print(f'{i} / {args.num_iters}')

        loss, loss_seg = trainer.train_step(unet_number=args.unet_number, max_batch_size=args.max_batch_size)
        
        if trainer.is_main:
            log = {f'{args.model_type}_loss': loss, f'{args.model_type}_loss_seg': loss_seg}
            # if args.log_wandb and (i % args.print_every == 0 or i == 0):
            #     wandb.log(log, step=i)
            if i % args.log_every == 0 or i == 0:
                saved_images, saved_labels = trainer.sample(
                    texts=args.test_caption if args.condition_on_text else None,
                    cond_scale=3., batch_size=args.test_batch_size,
                    start_at_unet_number=args.unet_number, stop_at_unet_number=args.unet_number,
                    start_image_or_video=start_image_or_video, start_label_or_video=start_label_or_video,
                    lowres_sample_noise_level=0.0,)
                
                # saved_images: 
                saved_bin = saved_images[:,0].unsqueeze(1)
                saved_hor = saved_images[:,1].unsqueeze(1)
                saved_ver = saved_images[:,2].unsqueeze(1)
                
                saved_bin = torch.cat([saved_bin, saved_bin, saved_bin], dim=1)
                saved_hor = torch.cat([saved_hor, saved_hor, saved_hor], dim=1)
                saved_ver = torch.cat([saved_ver, saved_ver, saved_ver], dim=1)

                # saved_labels = transform_lbl(saved_labels, 'train_id')
                # saved_images_labels = torchvision.utils.make_grid(
                #     torch.cat([saved_bin, saved_hor, saved_ver, saved_labels]), 
                #     nrow=max(4, args.test_batch_size), pad_value=1.)
                    # torch.cat([saved_images, saved_labels]), nrow=max(2, args.test_batch_size), pad_value=1.)
                # if args.log_wandb:
                #     wandb.log({**log, f'{args.model_type}_samples': wandb.Image(saved_images_labels,
                #               caption=args.sample_caption if args.condition_on_text else None)}, step=i)
                
                # del saved_images, saved_labels

    if trainer.is_main and args.log_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
