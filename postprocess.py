import numpy as np
from PIL import Image
import os
import glob
from model import FloorPlanNet  # Assuming we'll use some model functions
from predict import ind2rgb  # From our predict.py


def post_process(input_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    for path in input_paths:
        # Load prediction
        im = np.array(Image.open(path))
        im_ind = rgb2ind(im)  # Need to implement rgb2ind

        # Separate room and boundary
        rm_ind = im_ind.copy()
        rm_ind[im_ind == 9] = 0
        rm_ind[im_ind == 10] = 0

        bd_ind = np.zeros(im_ind.shape, dtype=np.uint8)
        bd_ind[im_ind == 9] = 9
        bd_ind[im_ind == 10] = 10

        # Post-processing steps
        hard_c = (bd_ind > 0).astype(np.uint8)
        cw_mask = fill_break_line(hard_c)
        fuse_mask = flood_fill(cw_mask + (rm_ind > 0).astype(np.uint8))
        new_rm_ind = refine_room_region(cw_mask, rm_ind)

        # Save result
        output = new_rm_ind.copy()
        output[bd_ind == 9] = 9
        output[bd_ind == 10] = 10
        rgb = ind2rgb(output)

        save_path = os.path.join(save_dir, os.path.basename(path))
        Image.fromarray(rgb).save(save_path)


# Implement the utility functions from util.py
def fill_break_line(cw_mask):
    # Implement similar to original but with OpenCV
    pass


def flood_fill(mask):
    # Implement flood fill
    pass


def refine_room_region(cw_mask, rm_ind):
    # Implement room refinement
    pass


def rgb2ind(rgb_im, color_map=floorplan_map):
    # Implement RGB to index conversion
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir',
                        type=str,
                        default='./out',
                        help='The folder that save network predictions.')
    args = parser.parse_args()

    input_dir = args.result_dir
    save_dir = os.path.join(input_dir, 'post')
    post_process(input_dir, save_dir)
