import os
import yaml
import argparse

import math
import torch

from tqdm import tqdm
from functools import partial
from torchvision import transforms
from torch.utils.data import DataLoader

import datasets
import models
import utils

import warnings
warnings.filterwarnings('ignore')


def batched_predict(model, inp, coord, scale, bsize):
    if coord is None:
        with torch.no_grad():
            pred = model(inp)
    else:
        with torch.no_grad():
            model.gen_feat(inp)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_rgb(coord[:, ql:qr, :], scale[:, ql:qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None,
              eval_bsize=None, verbose=False):
    if model is not None:
        model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
        
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    scale = None
    if eval_type is None:
        metric_fn = utils.calc_psnr
    else:
        dataset = eval_type.split('-')[0]
        # scale = int(eval_type.split('-')[1])
        scale = float(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset=dataset, scale=scale)

    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    
    output_idx = 1
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
            
        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['scale'])
        else:
            if scale is not None and scale > 4 and cell_decode:
                pred = batched_predict(model, inp, batch['coord'], batch['scale']*scale/4, eval_bsize)
            else:
                pred = batched_predict(model, inp, batch['coord'], batch['scale'], eval_bsize)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        b, c, ih, iw = batch['inp'].shape
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [b, round(ih * s), round(iw * s), c]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        batch['gt'] = batch['gt'].view(*shape).permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if eval_type is not None:
            data_case = eval_type.split('-')
            save_path = f'./output/{model_name}/{data_case[0].upper()}/{data_case[1]}x'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(pred.shape[0]):
                transforms.ToPILImage()(pred[i]).save(f'{save_path}/test_{output_idx:>03}.png')
                output_idx += 1
        
        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    global model_name
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    model_name = args.model.split('/')[-2]

    global cell_decode
    cell_decode = ('liif' in model_name) or ('lte' in model_name) or ('btc' in model_name)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'cell_decode': cell_decode})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8, pin_memory=True)
    
    res = eval_psnr(
                loader,
                model,
                data_norm = config.get('data_norm'),
                eval_type = config.get('eval_type'),
                eval_bsize = config.get('eval_bsize'),
                verbose = True,
            )
    
    print('result: {:.4f}'.format(res))