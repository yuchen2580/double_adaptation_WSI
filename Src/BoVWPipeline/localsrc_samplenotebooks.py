


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, Point
import sys
import os
import statistics
import psutil
import copy
import pickle
import re
from abc import ABC, abstractmethod
import math
import copy
import xml.etree.ElementTree as ET
import gc
from copy import deepcopy
from pathlib import Path
from skimage import data
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import skimage
import PIL
from skimage.filters import threshold_otsu
import torchvision.models as torchmodels


import torch.utils.data
import torchvision
import torch.nn as nn
list_pathstoadd = ["../../", "../../../PyDmed/",\
                   "../../../uda_and_microscopyimaging_repo2/Src/BoVWPipeline/"]
for path in list_pathstoadd:
    if(path not in sys.path):
        sys.path.append(path)
import pydmed
from pydmed.utils.data import *
import pydmed.lightdl
from pydmed.lightdl import *
import pydmed.stat
from pydmed.stat import *
import relatedwork
from relatedwork.utils.generativemodels import ResidualEncoder
import bovw_pipeline
from bovw_pipeline import *



class SamplePipeline1(BoVWPipeline):
    def __init__(self, num_classes, num_visualwords, device_stg3, *args, **kwargs):
        #grab privates  ====
        self.num_classes = num_classes
        self.num_visualwords = num_visualwords
        self.device_stg3 = device_stg3
        #call on super
        super(SamplePipeline1, self).__init__(*args, **kwargs)
        
    @abstractmethod
    def get_stg1_descriptorgenerator(self, size_input, kwargs_of_module):
        module_pretrained = relatedwork.utils.generativemodels.ResidualEncoder(
                                  resnettype=torchmodels.resnet50, pretrained=True
                                )
        #determine the size of output channel.
        with torch.no_grad():
            x = torch.randn(*size_input)
            size_pretrainedoutput = list(module_pretrained(x).size()) #[N,C,H,W]
        list_modules = list(module_pretrained.children())+\
               [nn.Conv2d(size_pretrainedoutput[1], 10, kernel_size=1, stride=1, padding=0)]
        module_stg1 = nn.Sequential(*list_modules)
        return module_stg1
    
    @abstractmethod
    def get_stg2_theencoderdictionary(self, size_input, kwargs_of_module):
        module_stg2 = FishVectEncoder(
                         size_input=size_input,
                         num_centers=self.num_visualwords,\
                         pi_of_fishvect=0.1, sigma_of_fishvect=0.1
                      )
        return module_stg2
    
    @abstractmethod
    def get_stg3_descriptorpooling(self, size_input, kwargs_of_module):
        module_stg3 = AvgPoolVectorsPerWSI(size_input=size_input,\
                                           device = self.device_stg3)
        return module_stg3
    
    
    @abstractmethod
    def get_stg4_finalclassifier(self, size_input, kwargs_of_module):
        N, M, _, _ = size_input
        module_stg4 = nn.Sequential(
                                    nn.Conv2d(
                                        M, self.num_classes,\
                                        kernel_size=1, stride=1, padding=0, bias=False
                                      )
                                )
        return module_stg4



