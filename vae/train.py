import os
import yaml
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

num_updates = 0

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_parser():
    parser = argparse.ArgumentParser(description='trainer for VAE models')
    parser.add_argument('--config',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file')
    return parser


def kl_anneal_func(cfg):
    if cfg['exp_params']['anneal_step'] == 0:
        return 1
    if cfg['exp_params']['anneal_func'] == 'logistic':
        return float(1 / (1 + np.exp(-cfg['exp_params']['sigmoid_factor'] * (num_updates - cfg['exp_params']['anneal_step']))))
    elif cfg['exp_params']['anneal_func'] == 'linear':
        return min(1, num_updates / cfg['exp_params']['anneal_step'])


def loss_fn(cfg, NLL, logp, target, length, kl, batch_size):
    kl_loss = kl.sum() / batch_size
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))
    nll_loss = NLL(logp, target) / batch_size
    kl_weight = kl_anneal_func(cfg)
    loss = nll_loss + kl_loss * kl_weight
    return loss, kl_loss, nll_loss, kl_weight


def train(cfg, dataloader, model, NLL, optimizer, epoch, logger):
    model.train()
    avg_loss = AverageMeter()
    avg_kl_loss = AverageMeter()
    avg_nll_loss = AverageMeter()
    for i, batch in enumerate(dataloader):
        region = batch['region'].cuda()
        input = batch['input'].cuda()
        target = batch['target'].cuda()
        length = batch['length'].cuda()
        kl, logp, _ = model(region, input, length)
        loss, kl_loss, nll_loss, kl_weight = loss_fn(cfg, NLL, logp, target, length, kl, len(input))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.update(loss)
        avg_kl_loss.update(kl_loss)
        avg_nll_loss.update(nll_loss)
        global num_updates

        if i % cfg['exp_params']['print_freq'] == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'KL {kl.val:.5f} ({kl.avg:.5f})\t' \
                'NLL {nll.val:.5f} ({nll.avg:.5f})\t' \
                'LR {lr:.7f}\t num_updates {num}\t kl_weight {weight}'.format(
                    epoch, i, len(dataloader), loss=avg_loss,
                    kl=avg_kl_loss, nll=avg_nll_loss, lr=optimizer.param_groups[0]['lr'], num=num_updates, weight=kl_weight
                )
            logger.info(msg)
        
        num_updates += 1


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
        os.makedirs(dest_dir, exist_ok=False)
    log_file = os.path.join(dest_dir, 'log.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(cfg)

    dataset = PhraseRegion(
        phrase_path=cfg['exp_params']['phrase_path'],
        region_path=cfg['exp_params']['region_path'],
        max_sequence_length=cfg['model_params']['max_sequence_length'],
        data_file=cfg['exp_params']['data_file'],
        vocab_file=cfg['exp_params']['vocab_file']
    )

    model = VAEModel(
        input_size=cfg['model_params']['region_embedding_size'],
        max_sequence_length=cfg['model_params']['max_sequence_length'],
        bos_idx=dataset.bos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        rnn_type=cfg['model_params']['rnn_type'],
        word_dropout=cfg['exp_params']['word_dropout'],
        embedding_dropout=cfg['exp_params']['embedding_dropout'],
        vocab_size=dataset.vocab_size,
        embedding_size=cfg['model_params']['embedding_size'],
        hidden_size=cfg['model_params']['hidden_size'],
        latent_size=cfg['model_params']['latent_size']
    ).cuda()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['exp_params']['batch_size'],
        shuffle=True,
        num_workers=20
    )

    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg['exp_params']['LR'], 
        weight_decay=cfg['exp_params']['weight_decay']
    )

    NLL = torch.nn.NLLLoss(ignore_index=dataset.pad_idx, reduction='sum')
    
    for epoch in range(cfg['exp_params']['max_epoches']):
        train(cfg, dataloader, model, NLL, optimizer, epoch, logger)
    
    torch.save(model.state_dict(), os.path.join(dest_dir, 'model_last.pth'))


if __name__ == '__main__':
    main()
