import os
import os.path
import re
import shutil
from glob import glob
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import NamedTuple, Callable
from PIL import Image
from PIL.Image import Image as img
from torch.utils.data import Dataset
import pydicom
from openslide import OpenSlide

from endaaman import Commander, curry, with_log

Image.MAX_IMAGE_PIXELS = 1000000000

J = os.path.join

with_wrote = lambda s: with_log(s, 'wrote {}')


class C(Commander):
    def run_save(self):
        o = OpenSlide('data/SFT/19-0295_HE.ndpi')
        for level, d in enumerate(o.level_dimensions):
            if level < 2:
                print(f'skip {level} {d}')
                continue
            i = o.get_thumbnail(d)
            i.save(f'tmp/wsi_{level}.jpg')
            print(f'save {level} {d}')

    def arg_thres(self, parser):
        parser.add_argument('--file', '-f', required=True)
        parser.add_argument('--level', '-l', type=int, default=3)
        parser.add_argument('--dilate', '-d', type=int, default=10)
        parser.add_argument('--skip', '-s', action='store_true')

    def run_thres(self):
        o = OpenSlide(self.args.file)
        name = os.path.splitext(os.path.basename(self.args.file))[0]
        dim = o.level_dimensions[self.args.level]
        original = o.get_thumbnail(dim)

        out_dir = with_log(f'out/crop/{name}', 'mkdir {}')
        os.makedirs(out_dir, exist_ok=True)

        wsi = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(with_wrote(J(out_dir, f'{self.args.level}_original.jpg')), wsi)
        if self.args.skip:
            print('ok:', ok)
            print('save original and abort.')
            return

        gray = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 20)

        cv2.imwrite(with_wrote(J(out_dir, f'{self.args.level}_thres.jpg')), thres)

        krn = np.ones((self.args.dilate, self.args.dilate), np.uint8)
        dialated = cv2.dilate(cv2.bitwise_not(thres), kernel=krn, iterations=1)
        cv2.imwrite(with_wrote(J(out_dir, f'{self.args.level}_dialated.jpg')), dialated)

        # contours, hierarchy = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.RET

        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                # remove small objects(b
                if cv2.contourArea(contours[i]) < (min(dim) / 20) ** 2:
                    continue
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                cv2.rectangle(wsi, (x, y), (x + w, y + h), (0, 0, 0), 10)

        cv2.imwrite(with_wrote(J(out_dir, f'{self.args.level}_result.jpg')), wsi)

        self.contours, self.hierarchy = contours, hierarchy
        self.wsi = wsi


c = C()
c.run()
