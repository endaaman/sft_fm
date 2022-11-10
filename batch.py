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

from endaaman import Commander, curry, with_wrote

Image.MAX_IMAGE_PIXELS = 1000000000


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
        parser.add_argument('--level', '-l', type=int, default=3)

    def run_thres(self):
        wsi = cv2.imread(f'tmp/wsi_{self.args.level}.jpg')
        gray = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 20)

        cv2.imwrite(with_wrote(f'tmp/thres_{self.args.level}.jpg'), thres)

        krn = np.ones((10, 10), np.uint8)
        dialated = cv2.dilate(cv2.bitwise_not(thres), kernel=krn, iterations=1)
        cv2.imwrite(with_wrote(f'tmp/dialated_{self.args.level}.jpg'), dialated)

        # contours, hierarchy = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.RET

        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                # remove small objects
                if cv2.contourArea(contours[i]) < 100*100:
                    continue
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                cv2.rectangle(wsi, (x, y), (x + w, y + h), (0, 0, 0), 10)

        cv2.imwrite(with_wrote(f'tmp/result_{self.args.level}.jpg'), wsi)

        self.contours, self.hierarchy = contours, hierarchy
        self.wsi = wsi


c = C()
c.run()
