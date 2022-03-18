import os
import yaml
import json
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname
from vae_model import VAEModel
from phrase_region_dataset import PhraseRegion

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser(description='trainer for VAE models')
    parser.add_argument('--config',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    filename = args.filename.split('/')[-1].split('.')[0]
    dest_dir = os.path.join(cfg['exp_params']['log_dir'], filename)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    region_embedding = torch.tensor(np.load(cfg['exp_params']['region_path']))
    with open(cfg['exp_params']['data_file'], 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(cfg['exp_params']['vocab_file'], 'r', encoding='utf8') as f:
        vocab = json.load(f)
    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = VAEModel(
        input_size=cfg['model_params']['region_embedding_size'],
        max_sequence_length=cfg['model_params']['max_sequence_length'],
        bos_idx=w2i['<s>'],
        eos_idx=w2i['</s>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        rnn_type=cfg['model_params']['rnn_type'],
        word_dropout=cfg['exp_params']['word_dropout'],
        embedding_dropout=cfg['exp_params']['embedding_dropout'],
        vocab_size=len(w2i),
        embedding_size=cfg['model_params']['embedding_size'],
        hidden_size=cfg['model_params']['hidden_size'],
        latent_size=cfg['model_params']['latent_size']
    ).cuda()
    
    model.load_state_dict(torch.load(join(dest_dir, 'model_last.pth')))
    model.eval()
    weight = model.embedding.weight.cpu().detach().numpy()
    np.save(join(dest_dir, 'embed_matrix.npy'), weight)
    latent_embedding = []
    with torch.no_grad():
        for idx in range(region_embedding.shape[0]):
            print('{0}/{1}'.format(idx, region_embedding.shape[0]))
            region = region_embedding[idx, :].unsqueeze(0).cuda()
            input = torch.tensor(data[str(idx)]['input']).unsqueeze(0).cuda()
            target = torch.tensor(data[str(idx)]['target']).unsqueeze(0).cuda()
            length = torch.tensor(data[str(idx)]['length']).unsqueeze(0).cuda()
            _, _, sampled_z = model(region, input, length)
            latent_embedding.append(sampled_z.squeeze(0).cpu().numpy())
    latent_embedding = np.array(latent_embedding)
    np.save(join(dest_dir, 'latent_embedding.npy'), latent_embedding)


if __name__ == '__main__':
    main()