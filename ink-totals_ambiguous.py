"""
This script finds the amount of black "ink" in each of the ambiguous sketches
generated in generate_candidates_ambiguous.py and adds it to the .npz file
for that category pair.
"""

import sys
import os
import json
import glob
import numpy as np
sketch_morph_path = os.getcwd()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from matplotlib.path import Path

# set numpy output to something sensible
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
sketch_dict = {}
categories = []
for file in glob.glob(os.path.join(sketch_morph_path, 'stimuli','candidates_ambiguous', '*.npz')):
    file_npz = np.load(file)
    category = file.split('/')[-1].split('.')[0]
    categories.append(category)
    sketch_dict[category] = file_npz['sketches']

for category in categories:
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('sketches loaded for {}'.format(category))
    print('Extracting information...')

    sketches = sketch_dict[category]
    ink_totals = []
    for kSketch in range(len(sketches)):
        # draw the sketch and save
        sketch = sketches[kSketch]
        fig, ax = plt.subplots(nrows=1,ncols=1,
                               figsize=(600/my_dpi,600/my_dpi))
        draw(sketch, ax=ax)
        plt.axis('off')
        plt.savefig(os.path.join(sketch_morph_path, 'stimuli',
                                 'candidates_ambiguous','images',
                                 category+'_temp.png'),dpi=my_dpi)

        im = Image.open(os.path.join(sketch_morph_path, 'stimuli',
                                     'candidates_ambiguous','images',
                                     category+'_temp.png')).convert('L');
        im_array = np.array(im)
        ink_total = sum(sum(1 - im_array/255))
        ink_totals.append(ink_total)
        im.close()
        if kSketch%15 == 0: plt.close('all') # limit number of open figures

    print('Saving info for {}'.format(category))

    with open(os.path.join(sketch_morph_path,'stimuli','candidates_ambiguous',
                           category+'_info.txt'), 'w') as f:
        f.write('mean ink total: {}'.format(np.mean(ink_totals))+'\n')
        f.write('standard deviation: {}'.format(np.std(ink_totals))+'\n')
        f.write('median: {}'.format(np.median(ink_totals))+'\n')
        f.write('max: {}'.format(np.max(ink_totals))+'\n')
        f.write('min: {}'.format(np.min(ink_totals))+'\n')
        f.close();

    print('Saving figure for {}'.format(category))

    # draw all 2000 sketches (why not?)
    fig, ax_arr = plt.subplots(nrows=200,
                               ncols=10,
                               figsize=(10, 200),
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
                                     'candidates_ambiguous',
                                     category+'.svg'), format='svg')

    np.savez(os.path.join(sketch_morph_path, 'stimuli', 'candidates_ambiguous', category+'.npz'),
             sketches=sketches,pix_count=ink_totals)

    plt.close('all')
