import os
import json
import torch
import numpy as np
import torch.nn as nn

from os.path import join, abspath, dirname
from rencoder import REncoder

root = dirname(dirname(abspath(__file__)))
dataset = join(root, 'dataset/multi30k/data/task1/tok')
corpus = join(dataset, 'train.lc.norm.tok.en')
nps = join(root, 'visual_grounding/files/np.json')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = REncoder(arch='bert', layers=[-4, -3, -2, -1], device=device).to(device)
    model.eval()
    with open(nps, 'r') as f:
        data = json.load(f)
        phrases = [np['phrase'] for item in data for np in item['nps']]
    phrase_embedding = []
    for index, phrase in enumerate(phrases):
        print('{0}/{1}'.format(index, len(phrases)))
        with torch.no_grad():
            tokens, final_state = model(phrase)
        embedding = torch.mean(final_state[1:-1], 0)
        norm = torch.norm(embedding)
        embedding /= norm
        phrase_embedding.append(embedding.cpu().numpy())
    
    phrase_embedding = np.array(phrase_embedding)
    print(phrase_embedding.shape)
    np.save(join(root, 'retrieve', 'phrase_embedding.npy'), phrase_embedding)


if __name__ == '__main__':
    main()
