import os
import json
import torch
import spacy
import numpy as np
import torch.nn as nn
from retrieve.rencoder import REncoder
from os.path import join, abspath, dirname

root = dirname(dirname(abspath(__file__)))


class TopkRetriever:
    def __init__(self, model, device='cpu'):
        self.model = model
        emb = join(root, 'retrieve/phrase_embedding.npy')
        nps = join(root, 'visual_grounding/files/np.json')
        self.phrase_embedding = torch.tensor(np.load(emb)).to(device)
        with open(nps, 'r') as f:
            data = json.load(f)
            self.phrase_table = [p['phrase'] for item in data for p in item['nps']]
        self.model.eval()

    def query(self, phrase, sentence=None, k=5):
        with torch.no_grad():
            tokens, final_state = self.model(phrase)
            embedding = torch.mean(final_state[1:-1, :], 0)
        norm = torch.norm(embedding)
        embedding /= norm
        if torch.isnan(torch.sum(embedding)):
            return None, None, None
        score = embedding.mul(self.phrase_embedding)
        score = torch.sum(score, -1)
        values, indices = torch.topk(score, k)
        return values, indices, [self.phrase_table[idx.item()] for idx in indices]


if __name__ == '__main__':
    main()
