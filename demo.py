import os
import argparse
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    scale_max = 4 # Maximum scale factor during training
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    h = int(img.shape[-2] * args.scale)
    w = int(img.shape[-1] * args.scale)
    scale = h / img.shape[-2]
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/scale_max, 1)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell_factor*cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
