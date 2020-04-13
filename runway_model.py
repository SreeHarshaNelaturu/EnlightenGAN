import time
import os
from options.test_options import TestOptions
from models.single_model import *
from util.visualizer import Visualizer
from PIL import Image
from data.base_dataset import get_transform
import torch
import runway
from runway.data_types import *

@runway.setup(options={"checkpoint" : file(extension=".pth"), "vgg_weights":file(extension=".weight")})
def setup(opts):
    checkpoint = opts["checkpoint"]
    vgg_weights = opts["vgg_weights"]

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
    opt.isTrain = False

    model = SingleModel()
    model.initialize(opt, checkpoint, vgg_weights)

    return {"model" : model,
            "opt" : opt}


command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}

@runway.command("enlighten_image", inputs=command_inputs, outputs=command_outputs, description="Adjust lightining on image")
def enlighten_image(model, inputs):

    infer_model = model["model"]
    
    transform = get_transform(model["opt"])
    A_img = inputs["input_image"].convert("RGB")

    A_img = A_img.resize((A_img.size[0]//16*16, A_img.size[1]//16*16), Image.BICUBIC)

    A_img = transform(A_img)

    r, g, b = A_img[0]+1, A_img[1]+1, A_img[2]+1
    A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.

    A_gray = torch.unsqueeze(torch.unsqueeze(A_gray, 0), 0)

    A_img = torch.unsqueeze(A_img, 0)

    img_dict = {"A" : A_img, "B" : A_img, "input_img" : A_img,  "A_gray" : A_gray}

    model["model"].set_input(img_dict)

    visuals = model["model"].predict()
    
    return {"output_image" : visuals["fake_B"]}



if __name__ == "__main__":
    runway.run()
