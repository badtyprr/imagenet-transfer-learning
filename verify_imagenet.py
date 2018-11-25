# Verifies the integrity of imagenet files and resizes/augments them

import argparse
import os
import pathlib
import logging

import cv2
import requests
from requests.exceptions import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify ImageNet integrity')
    parser.add_argument('--input', '-i', default='images/', type=pathlib.Path)
    parser.add_argument('--urls', '-u', default='list/urllist.txt', type=pathlib.Path)
    args = parser.parse_args()

    if not (args.input.exists() or args.urls.exists()):
        raise FileNotFoundError

    exceptions = (ConnectionError, ReadTimeout)

    files2urls = {}

    with open(args.urls) as inFile:
        for row in inFile.readlines():
            elements = row.split(' ')
            elements[1] = elements[1].replace("\"\"\"", '').replace('\n', '')
            files2urls[elements[0]] = elements[1]

    for file in os.listdir(args.input):
        file = os.path.join(args.input, file)
        image = cv2.imread(file)
        if image is None:
            logging.info('Redownloading {}'.format(file))
            try:
                r = requests.get(files2urls[file], timeout=30)
                f = open(file, "wb")
                f.write(r.content)
                f.close()
            except exceptions:
                continue
            # Verify once more
            image = cv2.imread(file)
            if image is None:
                logging.info("Download failed, removing {} [DRY RUN]".format(elements[0]))
                # os.remove(elements[0])
        else:
            logging.info('File {} is okay'.format(elements[0]))
            r = requests.get(elements[1], timeout=30)
            pass
