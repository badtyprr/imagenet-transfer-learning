# Detects image duplicates in a directory, and optionally, subdirectory of images.

# Python Libraries
import argparse
import pathlib
import os
import pickle
import logging
# 3rd Party Libraries
import cv2
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def quantized_histogram(image, bits=2):
    # Decimate to 2-bits
    image = cv2.divide(image, int(pow(2, 8-bits)))
    hist = np.zeros(pow(2,bits*3), dtype=np.uint)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            index = (image[row, col, 0] << 4) + (image[row, col, 1] << 2) \
                + (image[row, col, 2]) - 1
            hist[index] += 1
    return hist

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help='base directory to begin processing images',
                    default='.', type=pathlib.Path)
    ap.add_argument('-o', '--output',
                    help='output directory to write data file',
                    default='.', type=pathlib.Path)
    ap.add_argument('-d', '--dryrun',
                    help='do not delete files, just print out duplicates',
                    action='store_true')
    args = ap.parse_args()

    # Begin evaluating files
    # Data structure => dict[(filesize, histogram[0...63])],
    #   where histogram is a tuple of a quantized 6-bit RGB space (RRGGBB)
    d = {}
    output_filepath = os.path.join(args.output, "duplicates.bin")
    # Check for duplicate data
    if os.path.exists(output_filepath):
        logging.info('Duplicate data already exists: {}'.format(output_filepath))
        with open(output_filepath, 'rb') as pickle_fp_in:
            d = pickle.load(pickle_fp_in)
    else:
        logging.info('No duplicate data exists, searching image directory...')
        for root, dirs, files in os.walk(args.input):
            if not files:
                continue
            for file in files:
                filepath = os.path.join(root, file)
                # Read file stats (size)
                stats = os.stat(filepath)
                # Read image file
                image = cv2.imread(filepath, cv2.IMREAD_REDUCED_COLOR_8)
                # If the image cannot be opened, skip it
                if image is None:
                    continue
                hist = quantized_histogram(image)
                hashable_key = (stats.st_size, tuple(hist))
                try:
                    d[hashable_key].append(filepath)
                except KeyError:
                    d[hashable_key] = [filepath]

        # Write out the duplicates
        logging.info('Writing out duplicate data: {}'.format(output_filepath))
        with open(output_filepath, 'wb') as pickle_fp_out:
            pickle.dump(d, pickle_fp_out, protocol=pickle.HIGHEST_PROTOCOL)

    # Delete duplicates (keep the first one)
    for key in [k for k in d.keys() if len(d[k]) > 1]:
        for i in range(1,len(d[key])):
            logging.info('Removing duplicate: {}'.format(d[key][i]))
            try:
                if not args.dryrun:
                    os.remove(d[key][i])
            except FileNotFoundError:
                logging.warning('Could not find file: {}'.format(d[key][i]))

    pass
