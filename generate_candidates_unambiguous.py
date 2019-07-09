"""
This script generates the candidate sketches for unambiguous stimulus creation
in the sketch-morph project. It should be run in the "generate-candidates"
conda environment included in this repository
"""

import sys
import os
import json
import numpy as np
sketch_morph_path = os.getcwd()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# import numpy and set default output
import numpy as np
np.set_printoptions(precision=8,
                    edgeitems=6,
                    linewidth=200,
                    suppress=True)

from magenta.models.sketch_rnn.utils import *
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

# get list of morph pairs
morph_pairs = os.listdir(os.path.join(sketch_morph_path, 'hyper','datasets'))

# generate sketch info
for morph_pair in morph_pairs:
    categories = morph_pair.split('_')

    for category in categories:
        # load sketches from the test set
        sketches = np.load(os.path.join(sketch_morph_path, 'hyper', 'datasets',
                                          morph_pair, category+'.npz'),
                                          encoding='bytes')['test']

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Test sketches loaded for {}'.format(category))
        print('Extracting information...')

        ink_totals = []
        for kSketch in range(len(sketches)):
            # draw the sketch and save
            sketch = sketches[kSketch]
            fig, ax = plt.subplots(nrows=1,ncols=1,
                                   figsize=(600/my_dpi,600/my_dpi))
            draw(sketch, ax=ax)
            plt.axis('off')
            plt.savefig(os.path.join(sketch_morph_path, 'stimuli',
                                     'candidates_unambiguous','images',
                                     category+'_temp.png'),dpi=my_dpi)

            im = Image.open(os.path.join(sketch_morph_path, 'stimuli',
                                         'candidates_unambiguous','images',
                                         category+'_temp.png')).convert('L');
            im_array = np.array(im)
            ink_total = sum(sum(1 - im_array/255))
            ink_totals.append(ink_total)
            im.close()
            if kSketch%15 == 0: plt.close('all')

        print('Saving info for {}'.format(category))

        with open(os.path.join(sketch_morph_path,'stimuli','candidates_unambiguous',
                               category+'_info.txt'), 'w') as f:
            f.write('mean ink total: {}'.format(np.mean(ink_totals))+'\n')
            f.write('standard deviation: {}'.format(np.std(ink_totals))+'\n')
            f.write('median: {}'.format(np.median(ink_totals))+'\n')
            f.write('max: {}'.format(np.max(ink_totals))+'\n')
            f.write('min: {}'.format(np.min(ink_totals))+'\n')
            f.close();

        print('Saving figure for {}'.format(category))

        # draw all 5000 sketches (why not?)
        fig, ax_arr = plt.subplots(nrows=100,
                                   ncols=25,
                                   figsize=(25, 100),
                                   subplot_kw=dict(xticks=[],
                                                   yticks=[],
                                                   frame_on=False))
        im_no = 0
        for ax_row in ax_arr:
            for ax in ax_row:
                draw(sketches[im_no], ax=ax)
                ax.set_xlabel(round(ink_totals[im_no],2), fontsize=8)
                im_no+=1

        plt.savefig(os.path.join(sketch_morph_path, 'stimuli',
                                         'candidates_unambiguous',
                                         category+'.svg'), format='svg')

        np.savez(os.path.join(sketch_morph_path, 'stimuli', 'candidates_unambiguous', category+'.npz'),
                 sketches=sketches,pix_count=ink_totals)

        plt.close('all')
