import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

    # save image to the disk
    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % ("lol", label)
            save_path = os.path.join("./", image_name)
            util.save_image(image_numpy, save_path)
            print("Kawaii")
        #webpage.add_images(ims, txts, links, width=self.win_size)
