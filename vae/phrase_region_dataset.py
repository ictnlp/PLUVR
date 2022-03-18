import io
import os
import json
import numpy as np
import torch
from collections import defaultdict, Counter, OrderedDict
from torch.utils.data import Dataset

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class PhraseRegion(Dataset):
    def __init__(self, phrase_path, region_path, max_sequence_length, data_file, vocab_file):
        self.region_embedding = torch.tensor(np.load(region_path))
        self.raw_data_path = phrase_path
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.max_sequence_length = max_sequence_length

        if os.path.exists(self.data_file) and os.path.exists(self.vocab_file):
            self._load_data()
        else:
            self.data_file = 'phrase.json'
            self.vocab_file = 'vocab.json'
            self._create_data()
    
    def __getitem__(self, index):
        return {
            'region': self.region_embedding[index],
            'input': torch.tensor(self.data[str(index)]['input']),
            'target': torch.tensor(self.data[str(index)]['target']),
            'length': torch.tensor(self.data[str(index)]['length'])
        }
    
    def __len__(self):
        return len(self.data)
    
    @property
    def vocab_size(self):
        return len(self.w2i)
    
    @property
    def pad_idx(self):
        return self.w2i['<pad>']
    
    @property
    def bos_idx(self):
        return self.w2i['<s>']
    
    @property
    def eos_idx(self):
        return self.w2i['</s>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']
    
    def get_w2i(self):
        return self.w2i
    
    def get_i2w(self):
        return self.i2w
    
    def _load_data(self, vocab=True):
        print('loading data...')
        with open(self.data_file, 'r', encoding='utf8') as f:
            self.data = json.load(f)
        if vocab:
            self._load_vocab()
    
    def _load_vocab(self):
        print('loading vocab...')
        with open(self.vocab_file, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
    
    def _create_data(self):
        print('creating data...')
        self._create_vocab()
        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as f:
            for i, line in enumerate(f):
                words = line.strip().split()
                input = ['<s>'] + words
                input = input[:self.max_sequence_length]
                target = words[:self.max_sequence_length-1]
                target = target + ['</s>']
                assert len(input) == len(target)
                length = len(input)
                input.extend(['<pad>'] * (self.max_sequence_length - length))
                target.extend(['<pad>'] * (self.max_sequence_length - length))
                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]
                idx = len(data)
                data[idx]['input'] = input
                data[idx]['target'] = target
                data[idx]['length'] = length
        
        with io.open(self.data_file, 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))
        self._load_data(vocab=False)
    
    def _create_vocab(self):
        print('creating vocab...')
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        special_tokens = ['<s>', '<pad>', '</s>', '<unk>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
        with open(self.raw_data_path, 'r') as f:
            for i, line in enumerate(f):
                words = line.strip().split()
                w2c.update(words)
            for w, c in w2c.items():
                if w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        with open(self.vocab_file, 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))
        self._load_vocab()