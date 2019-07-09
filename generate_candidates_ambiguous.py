"""
This script generates the candidate sketches for ambiguous stimulus creation
in the sketch-morph project. It should be run in the "generate-candidates"
conda environment included in this repository. The script implements the
tensorflow model trained over the training data from each pair of categories,
uses an SVM classifier to find a hyperplane separating the latent
representations of the sketches in the test data, then generates new exemplars
that are orthogonal to this hyperplane in the latent space of the model.
"""

import sys
import os
import json
sketch_morph_path = os.getcwd()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import numpy.linalg as la
import tensorflow as tf

from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.path import Path
from matplotlib import rc

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from itertools import product
from six.moves import map, zip

# set numpy output to something sensible
np.set_printoptions(precision=8,
                    edgeitems=6,
                    linewidth=200,
                    suppress=True)

# get all of the command line tools from magenta
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from PIL import Image
my_dpi = 192


# define some classes and functions for making stimuli
class SketchPath(Path):

    def __init__(self, data, factor=.2, *args, **kwargs):

        vertices = np.cumsum(data[::, :-1], axis=0) / factor
        codes = np.roll(self.to_code(data[::,-1].astype(int)),
                        shift=1)
        codes[0] = Path.MOVETO

        super(SketchPath, self).__init__(vertices,
                                         codes,
                                         *args,
                                         **kwargs)

    @staticmethod
    def to_code(cmd):
        # if cmd == 0, the code is LINETO
        # if cmd == 1, the code is MOVETO (which is LINETO - 1)
        return Path.LINETO - cmd


def draw(sketch_data, factor=.2, pad=(10, 10), ax=None):

    if ax is None:
        ax = plt.gca()

    x_pad, y_pad = pad

    x_pad //= 2
    y_pad //= 2

    x_min, x_max, y_min, y_max = get_bounds(data=sketch_data,
                                            factor=factor)

    ax.set_xlim(x_min-x_pad, x_max+x_pad)
    ax.set_ylim(y_max+y_pad, y_min-y_pad)

    sketch = SketchPath(sketch_data)

    patch = patches.PathPatch(sketch, facecolor=(1.,1.,1.,0.))
    ax.add_patch(patch)

def encode(input_strokes, max_len):
    strokes = to_big_strokes(input_strokes, max_len).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    z = sess.run(eval_model.batch_z,
                 feed_dict={
                    eval_model.input_data: [strokes],
                    eval_model.sequence_lengths: seq_len})[0]
    return z

def decode(z_input=None, temperature=.1, factor=.2):
    z = None
    if z_input is not None:
        z = [z_input]
    sample_strokes, m = sample(
        sess,
        sample_model,
        seq_len=eval_model.hps.max_seq_len,
        temperature=temperature, z=z)
    return to_normal_strokes(sample_strokes)

# function for getting orthogonal vectors
def orthogonalize(U, eps=1e-50):
    n = len(U[0])
    V = U
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if la.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= la.norm(V[i])
    return V

# morph_pairs = os.listdir(os.path.join(sketch_morph_path, 'hyper','models'))

# Get seeds for svm
seed_dict = {'face_radio': [150, 50],
             'pig_alarm-clock': [132, 32]}

## Seeds for other pairs not included in this repo
# 'face_strawberry': [147, 47],
# 'foot_hockey-stick': [136, 36],
# 'hand_cactus': [148, 48],
# 'hedgehog_bush': [134, 34],
# 'lion_sun': [133, 33],
# 'rabbit_scissors': [128, 28],

for morph_pair in seed_dict.keys():
    # implement model for this pair
    nn_arch = 'hyper_lstm'
    DATA_DIR = os.path.join(sketch_morph_path,'hyper','datasets', morph_pair)
    MODELS_ROOT_DIR = os.path.join(sketch_morph_path, 'hyper', 'models')
    MODEL_DIR = os.path.join(MODELS_ROOT_DIR, morph_pair, nn_arch)

    (train_set,
     valid_set,
     test_set,
     hps_model,
     eval_hps_model,
     sample_hps_model) = load_env(DATA_DIR, MODEL_DIR)

    reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    load_checkpoint(sess=sess, checkpoint_path=MODEL_DIR)

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Done loading model for {}'.format(morph_pair))

    # get latent vectors of test sketches
    Z_svm_neg = np.load(os.path.join(sketch_morph_path, 'stimuli', 'latent_vectors',
                                 morph_pair.split('_')[0]+'.npz'))['latent_vectors']
    Z_svm_pos = np.load(os.path.join(sketch_morph_path, 'stimuli', 'latent_vectors',
                                 morph_pair.split('_')[1]+'.npz'))['latent_vectors']

    Z_svm_all = np.vstack((Z_svm_neg,Z_svm_pos))

    print('Building SVC for {}'.format(morph_pair))

    # get test strokes into array and create target vector
    targs = np.hstack((np.repeat(-1,2500), np.repeat(1,2500)))

    # split into training and testing sets. also shuffles the data
    X_train, X_test, y_train, y_test = train_test_split(Z_svm_all,
                                                        targs,
                                                        test_size=0.20,
                                                        random_state=seed_dict[morph_pair][1])

    # clear the old variables if a model has already been run
    if 'clf' in vars():
        del clf

    clf = svm.LinearSVC(C=1.0,fit_intercept=False, max_iter=2000000,
                        random_state=seed_dict[morph_pair][0])
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)

    print('Saving SVC results'.format(morph_pair))

    with open(os.path.join(sketch_morph_path,'stimuli','sketches','SVM_info','{}.txt'.format(morph_pair)), 'w') as f:
        f.write(str(clf)+'\n\n')
        f.write('F1_train: '+str(f1_score(y_train, clf.predict(X_train), average='macro'))+'\n\n')
        f.write('Acc_train: '+str(clf.score(X_train,y_train))+'\n\n')
        f.write('F1_test: '+str(f1_score(y_test,preds, average='macro'))+'\n\n')
        f.write('Acc_test: '+str(clf.score(X_test,y_test))+'\n\n')
        f.write('classification_report: \n'+str(report)+'\n\n')
        f.write('confusion_matrix: \n'+str(confusion_matrix(y_test,preds))+'\n\n')
        f.close();

    candidates = []
    for qMult in [1, 3, 5, 8]:
        # create orthonormal basis set
        coefs = clf.coef_
        coefs = np.reshape(coefs,(128,))
        norm_coefs = coefs/la.norm(coefs)
        norm_coefs = np.reshape(norm_coefs,(128,))

        X = []
        [X.append(np.random.random(size=(128,))) for i in range(128)];
        X[0] = coefs

        Q = qMult*orthogonalize(np.array(X))

        for decodeTemp in [.05, .25, .45, .65, .85]:
            print('Saving Sketches for {}, factor = {}, temperature = {}'.format(morph_pair, qMult, decodeTemp))
            [candidates.append(decode(Q[Q_idx+1], temperature=decodeTemp)) for Q_idx in range(100)];

    np.savez(os.path.join(sketch_morph_path, 'stimuli', 'candidates_ambiguous', morph_pair), sketches=candidates)
