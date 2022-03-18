import os
import json
import torch
import numpy as np
import torch.nn as nn
from transformers import *

LANG_MODELS = {
          'bert':    (BertModel,       BertTokenizer,       'bert-base-uncased'),
          'bert-large':  (BertModel,       BertTokenizer,       'bert-large-uncased'),
          'gpt':     (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          'gpt2':    (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          'ctrl':    (CTRLModel,       CTRLTokenizer,       'ctrl'),
          'xl':      (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          'xlnet':   (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          'xlm':     (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          'distil':  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          'roberta': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'xlm-roberta': (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
}


class REncoder(nn.Module):
    def __init__(self, arch, layers, device):
        super().__init__()
        Model, Tokenizer, weight = LANG_MODELS[arch]
        self.bert = Model.from_pretrained(
            weight,
            output_hidden_states=True
        )
        self.tokenizer = Tokenizer.from_pretrained(weight)
        self.layers = layers
        self.device = device
    
    def forward(self, sent):
        inputs = self.tokenizer(sent, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        output, pooled_output, hidden_states = self.bert(input_ids, token_type_ids, attention_mask)[:3]
        final_state = torch.cat(list(hidden_states[layer] for layer in self.layers), -1).squeeze(0)
        return tokens, final_state