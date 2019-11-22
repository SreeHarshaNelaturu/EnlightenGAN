import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from PIL import Image
from data.base_dataset import get_transform
import torch

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.instance_norm = 0
opt.resize_or_crop = 'no'
opt.norm = "instance"
opt.self_attention = True
opt.times_residual = True
opt.no_dropout = True


transform = get_transform(opt)
A_img = Image.open("nimmo.jpeg").convert('RGB')
B_img = A_img


A_size = A_img.size
A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
A_img = A_img.resize(A_size, Image.BICUBIC)

A_img = transform(A_img)
B_img = transform(B_img)

r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
A_gray = torch.unsqueeze(A_gray,0)
A_gray = torch.unsqueeze(A_gray, 0)
print("ladidia", A_gray.shape)

A_img = torch.unsqueeze(A_img, 0)
B_img = torch.unsqueeze(B_img, 0)


input_img = A_img

img_dict = {"A" : A_img, "B" : B_img, "input_img" : input_img,  "A_gray" : A_gray}
model = create_model(opt) 
visualizer = Visualizer(opt)

model.set_input(img_dict)
visuals = model.predict()
visualizer.save_images(visuals)




