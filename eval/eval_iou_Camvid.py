# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import camvid
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

from FSFNet_Camvid import FSFNet


NUM_CHANNELS = 3
NUM_CLASSES = 12

image_transform = ToPILImage()
input_transform_camvid = Compose([

    ToTensor(),
])
target_transform_camvid = Compose([
    ToLabel(),
    Relabel(255, 11),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = FSFNet(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            # print(name)
            # print(param)
            if name not in own_state:

                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(camvid(args.datadir, input_transform_camvid, target_transform_camvid, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images, volatile=True)
        outputs = model(inputs)


        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)



        filenameSave = filename[0].split("images/")[1]

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Sky")
    print(iou_classes_str[1], "Building")
    print(iou_classes_str[2], "Pole")
    print(iou_classes_str[3], "Road")
    print(iou_classes_str[4], "Pavement")
    print(iou_classes_str[5], "Tree")
    print(iou_classes_str[6], "SignSymbol")
    print(iou_classes_str[7], "Fence")
    print(iou_classes_str[8], "Car")
    print(iou_classes_str[9], "Pedestrian")
    print(iou_classes_str[10], "Bicyclist")

    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # diversenet 할 때 추가해줘야함   "0,1"
    parser = ArgumentParser()
    parser.add_argument('--state')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--loadDir', default="../save/Camvid/camvid_imagenet/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="FSFNet_Camvid.py")
    parser.add_argument('--subset', default="test")  #can be val, test, train, demoSequence
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/DB/camvid_kaggle/camvid/CamVid/")

    main(parser.parse_args())
