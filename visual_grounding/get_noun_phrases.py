import os
import json
import spacy

from os.path import join, abspath, dirname

root = dirname(dirname(abspath(__file__)))
dataset = join(root, 'dataset/multi30k/data/task1/tok')
data = join(dataset, 'train.lc.norm.tok.en')

nlp = spacy.load('en_core_web_sm')


def getNPs(sentence):
    doc = nlp(sentence)
    nps = []
    for chunk in doc.noun_chunks:
        nps.append({
            'phrase': chunk.text,
            'head': chunk.root.lemma_
        })
    return nps


def main():
    results = []
    with open(data, 'r') as f:
        corpus = f.readlines()
        for index, sentence in enumerate(corpus):
            nps = getNPs(sentence)
            result = {
                'index': index,
                'nps': nps
            }
            results.append(result)
            print('{0}/{1}: '.format(index, len(corpus)), nps)

    with open(join(root, 'visual_grounding/files/np.json'), 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4, separators=(',', ':'))


if __name__ == '__main__':
    main()