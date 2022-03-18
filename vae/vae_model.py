import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VAEBottleneck(nn.Module):
    def __init__(self, hidden_size, latent_size=None):
        super(VAEBottleneck, self).__init__()
        self.hidden_size = hidden_size
        if latent_size is None:
            self.latent_size = self.hidden_size
        else:
            self.latent_size = latent_size
        self.dense = nn.Linear(hidden_size, self.latent_size * 2)

    def forward(self, x, sampling=True, residual_q=None):
        vec = self.dense(x)
        mu = vec[:, :self.latent_size]
        if residual_q is not None:
            mu = 0.5 * (mu + residual_q[:, :self.latent_size])
        if not sampling:
            return mu, vec
        else:
            var = F.softplus(vec[:, self.latent_size:])
            if residual_q is not None:
                var = 0.5 * (var + F.softplus(residual_q[:, self.latent_size:]))
            noise = mu.clone()
            noise = noise.normal_()
            z = mu + noise * var
            return z, vec


class VAEModel(nn.Module):
    def __init__(self, input_size, max_sequence_length, bos_idx, eos_idx, pad_idx, unk_idx, rnn_type, word_dropout, embedding_dropout, vocab_size, embedding_size, hidden_size=512, latent_size=64):
        super(VAEModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # rnn setting
        self.max_sequence_length = max_sequence_length
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.word_dropout = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()
        self.encoder_rnn = rnn(embedding_size, hidden_size, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, batch_first=True)
        
        # transform v from input_size to hidden_size
        self.input_encoder = nn.Linear(input_size, hidden_size)
        # prior p(z|v)
        self.prior_prob_estimator = nn.Linear(hidden_size, latent_size * 2)
        # approximated posterior q(z|v,w)
        self.bottleneck = VAEBottleneck(hidden_size * 2, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.hidden2decinit = nn.Linear(hidden_size * 2, hidden_size)
        self.output2vocab = nn.Linear(hidden_size, vocab_size)
    
    def sample_from_posterior(self, posterior_states, sampling=True):
        sampled_z, posterior_prob = self.bottleneck(posterior_states, sampling=sampling)
        full_vector = self.latent2hidden(sampled_z)
        return full_vector, posterior_prob

    def compute_vae_kl(self, prior_prob, posterior_prob):
        mu1 = posterior_prob[:, :self.latent_size]
        var1 = F.softplus(posterior_prob[:, self.latent_size:])
        mu2 = prior_prob[:, :self.latent_size]
        var2 = F.softplus(prior_prob[:, self.latent_size:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
            (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def forward(self, image, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)
        hidden = hidden.squeeze(0)
        # dim transform for v
        image = self.input_encoder(image)
        # compute p(z|v)
        prior_prob = self.prior_prob_estimator(image)
        # compute q(z|v,w)
        sampled_z, posterior_prob = self.sample_from_posterior(torch.cat((image, hidden), -1))
        # decode p(w|v,z)
        hidden = self.hidden2decinit(torch.cat((image, sampled_z), -1))
        # apply dropout to decoder input sequence
        if self.word_dropout > 0:
            prob = torch.rand(input_sequence.size()).cuda()
            prob[(input_sequence.data - self.bos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden.unsqueeze(0))
        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()
        # project outputs to vocab
        logp = F.log_softmax(self.output2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        # calculate kl term
        kl = self.compute_vae_kl(prior_prob, posterior_prob)
        return kl, logp, hidden
