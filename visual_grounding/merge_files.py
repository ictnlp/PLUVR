import os
import numpy as np

from os.path import join, abspath, dirname

root = dirname(dirname(abspath(__file__)))
files = join(root, 'visual_grounding/files')

def main():
    all_features = []
    for i in range(11):
        path = join(files, 'bbox_features_faster_rcnn_R_101_' + ('%02d' % i) + '.npy')
        print('loading {0}/10...'.format(i))
        features = np.load(path)
        all_features.extend(features.tolist())
    all_features = np.array(all_features)
    print(all_features.shape)
    np.save(join(files, 'bbox_features_faster_rcnn_R_101.npy'), all_features)


if __name__ == '__main__':
    main()