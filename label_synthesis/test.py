import argparse
import os
import numpy as np
import os.path as osp

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from datasets.lizard import mapping_id as mapping_id_lizard
from datasets.lizard import transform_lbl as transform_lbl_lizard
from datasets.cityscapes import ToTensorNoNorm
from imagen_pytorch import BaseJointUnet, JointImagen, JointImagenTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, nargs='+', required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--save_img_path', type=str, required=True)

    parser.add_argument('--lowres_dir', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=7)

    parser.add_argument('--dataset', type=str, default='cityscapes')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--start_sample_idx', type=int, default=0, help='included')
    parser.add_argument('--end_sample_idx', type=int, default=2975, help='not included')
    parser.add_argument('--num_samples', type=int, default=0, help='included')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_captions', type=str, nargs='*', default=['', ])
    parser.add_argument('--caption_list_dir', type=str, default='')

    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--sample_timesteps', type=int, default=100)
    parser.add_argument('--cond_scale', type=float, nargs='+', default=(3.0, ))
    parser.add_argument('--lowres_sample_noise_level', type=float, default=0.2)
    parser.add_argument('--start_image_or_video', type=str,
                        default='samples/frankfurt_000000_000294_leftImg8bit.png')
    parser.add_argument('--start_label_or_video', type=str,
                        default='samples/frankfurt_000000_000294_gtFine_labelIds.png')
    parser.add_argument('--return_all_unet_outputs', action='store_true')
    parser.add_argument('--start_at_unet_number', type=int, default=1)
    parser.add_argument('--stop_at_unet_number', type=int, default=3)

    parser.add_argument('--noise_schedules', type=str, nargs='*', default=('cosine', ))
    parser.add_argument('--noise_schedules_lbl', type=str, nargs='*', default=('cosine_p', ))
    parser.add_argument('--cosine_p_lbl', type=float, default=1.0)

    parser.add_argument('--channels_lbl', type=int, default=3)
    parser.add_argument('--pred_objectives', type=str, default='noise')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--no_condition_on_text', action='store_false', dest='condition_on_text')
    parser.set_defaults(condition_on_text=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)

    args = parser.parse_args()

    args.cond_scale = args.cond_scale[0] if len(args.cond_scale) == 1 else args.cond_scale
    args.test_captions = [args.test_captions[0] + " nucleus area ratio"]
    
    if len(args.test_captions) == 1 and args.test_batch_size != 1:
        args.test_captions = args.test_captions * args.test_batch_size

    assert len(args.test_captions) == args.test_batch_size, \
        (len(args.test_captions), args.test_batch_size)

    args.end_sample_idx = args.start_sample_idx + args.num_samples
    
    print(f'Sample Indices: {args.start_sample_idx} - {args.end_sample_idx}')

    return args


def main():
    args = parse_args()

    # unet for imagen

    print(f'Creating JointUNets.. {args.model_type}')

    start_at_unet_number = args.start_at_unet_number
    stop_at_unet_number = args.stop_at_unet_number
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
        args.unet_number = 1
    else:
        raise NotImplementedError(args.model_type)

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = JointImagen(
        unets=unets,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        sample_timesteps=args.sample_timesteps,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text=args.condition_on_text,
        pred_objectives=args.pred_objectives,
        noise_schedules=args.noise_schedules,
        noise_schedules_lbl=args.noise_schedules_lbl,
        cosine_p_lbl=args.cosine_p_lbl,
    )
    trainer = JointImagenTrainer(
        imagen,
        fp16=args.fp16,
        dl_tuple_output_keywords_names=('images', 'labels', 'texts'),
    )
    trainer.load(args.checkpoint_path[0])

    if args.dataset == 'lizard':
        mapping_id = mapping_id_lizard
        transform_lbl = transform_lbl_lizard
    else:
        raise NotImplementedError(args.dataset)

    start_image_or_video, start_label_or_video = None, None

    n_idx = args.start_sample_idx
    while n_idx < args.end_sample_idx:
        assert args.save_path.endswith(('.png', ))
        os.makedirs(osp.dirname(args.save_path), exist_ok=True)
        
        os.makedirs(osp.dirname(args.save_img_path), exist_ok=True)
        os.makedirs(os.path.join(args.save_img_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(args.save_img_path, 'binary_labels'), exist_ok=True)
        os.makedirs(os.path.join(args.save_img_path, 'dists'), exist_ok=True)

        batch_size = 0
        texts = []
        for test_caption in args.test_captions:
            if n_idx >= args.end_sample_idx:
                break
            texts.append(test_caption)
            # texts = test_caption
            batch_size += 1
            n_idx += 1
        
        nucleus_proportion = texts[0].split(' ')[0]
        
        print(f'{n_idx} / {args.end_sample_idx}: {texts}')
        outputs = trainer.sample(
            texts=texts,
            cond_scale=args.cond_scale, batch_size=batch_size,
            start_at_unet_number=start_at_unet_number, stop_at_unet_number=stop_at_unet_number,
            start_image_or_video=start_image_or_video, start_label_or_video=start_label_or_video,
            lowres_sample_noise_level=args.lowres_sample_noise_level,
            return_all_unet_outputs=args.return_all_unet_outputs,
            use_tqdm=True)

        if not args.return_all_unet_outputs:
            outputs = [outputs]
        for idx_unet, output in enumerate(outputs):
            fn = os.path.basename(args.save_path.replace('base.png', f'{nucleus_proportion}_{n_idx-1}_{idx_unet}.png'))
            pth_lbl = os.path.join(args.save_img_path, 'labels', fn)
            pth_img = os.path.join(args.save_img_path, 'binary_labels', fn)
            pth_dis = os.path.join(args.save_img_path, 'dists', fn.split('.png')[0] + '.npy')

            saved_images, saved_labels = output

            # image 따로 저장
            # saved_images: 
            bin = (saved_images[:,0].squeeze(0).numpy() * 255).astype(np.uint8)
            hor = saved_images[:,1].numpy()
            ver = saved_images[:,2].numpy()
            hv_map = np.concatenate([hor, ver], axis=0).transpose(1,2,0)
            lbl = saved_labels.squeeze(0).squeeze(0).numpy().astype(np.uint8)

            np.save(pth_dis, hv_map)
            Image.fromarray(bin).save(pth_img)
            Image.fromarray(lbl).save(pth_lbl)

            saved_bin = saved_images[:,0].unsqueeze(1)
            saved_hor = saved_images[:,1].unsqueeze(1)
            saved_ver = saved_images[:,2].unsqueeze(1)
            
            saved_bin = torch.concatenate([saved_bin, saved_bin, saved_bin], dim=1)
            saved_hor = torch.concatenate([saved_hor, saved_hor, saved_hor], dim=1)
            saved_ver = torch.concatenate([saved_ver, saved_ver, saved_ver], dim=1)

            saved_labels = transform_lbl(saved_labels, 'train_id')

            torchvision.utils.save_image(
                torch.cat([saved_bin, saved_hor, saved_ver, saved_labels]),
                args.save_path.replace('base.png', f'{nucleus_proportion}_{n_idx-1}_{idx_unet}.png'),
                nrow=max(4, args.test_batch_size), pad_value=1.)
            print(args.save_path.replace('base.png', f'{nucleus_proportion}_{n_idx-1}_{idx_unet}.png') + ' has been saved.')


if __name__ == '__main__':
    main()