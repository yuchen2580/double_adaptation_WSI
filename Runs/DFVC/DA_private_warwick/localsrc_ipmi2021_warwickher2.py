


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



def visualize_one_patient(patient, list_smallchunks):
    '''
    Given all smallchunks collected for a specific patient, this function
    should visualize the patient. 
    Inputs:
        - patient: the patient under considerations, an instance of `utils.data.Patient`.
        - list_smallchunks: the list of all collected small chunks for the patient,
            a list whose elements are an instance of `lightdl.SmallChunk`.
    '''
    #settings =======
    scale_2 = 0.1 #=====
    fname_wsi = patient.dict_records["wsi"].rootdir + patient.dict_records["wsi"].relativedir
    opsimage = openslide.OpenSlide(fname_wsi)
    scale_1 = 1.0/opsimage.level_downsamples[1]
    W, H = opsimage.dimensions
    opsimageW, opsimageH = opsimage.dimensions
    W, H = int(W*scale_1*scale_2), int(H*scale_1*scale_2)
    pil_thumbnail = opsimage.get_thumbnail((W,H))
    plt.ioff()
    fig, ax = plt.subplots(1,2, figsize=(2*10,10))
    ax[0].imshow(pil_thumbnail)
    ax[0].axis('off')
    ax[0].set_title("patient {}, H&E [{} x {}]."\
                    .format(patient.int_uniqueid, opsimageW, opsimageH))
    ax = ax[1]
    ax.imshow(pil_thumbnail)
    ax.axis('off')
    print("patient {}, number of smallchunks={}"\
          .format(patient, len(list_smallchunks)))
    list_colors = ['lawngreen', 'cyan', 'gold', 'greenyellow']
    list_shownbigchunks = []
    for smallchunk in list_smallchunks:
        #show the bigchunk ================
        x = smallchunk.dict_info_of_bigchunk["x"]
        y = smallchunk.dict_info_of_bigchunk["y"]
        x, y = int(x*scale_1*scale_2), int(y*scale_1*scale_2)
        if(not([x,y] in list_shownbigchunks)):
            w, h = int(4000*scale_1*scale_2), int(4000*scale_1*scale_2)
            rect = patches.Rectangle((x,y), w, h, linewidth=1,\
                                      linestyle="--",\
                                      edgecolor=random.choice(list_colors),\
                                      facecolor='none', fill=False)
            ax.add_patch(rect)
            list_shownbigchunks.append([x,y])
        
        #get x,y,w,h ======
        x = smallchunk.dict_info_of_smallchunk["x"]*scale_2 +\
            smallchunk.dict_info_of_bigchunk["x"]*scale_1*scale_2
        y = smallchunk.dict_info_of_smallchunk["y"]*scale_2 +\
            smallchunk.dict_info_of_bigchunk["y"]*scale_1*scale_2
        x, y = int(x), int(y)
        w, h = int(224*4*scale_1*scale_2), int(224*4*scale_1*scale_2)
        x_centre, y_centre = int(x+0.5*w), int(y+0.5*h)
        #make-show the rect =====
        circle = patches.Circle((x_centre, y_centre), radius=w*0.05,\
                                 facecolor=random.choice(list_colors),\
                                 fill=True)
        ax.add_patch(circle)
    #show the annotations (if the patient has annotations) =========
    if(flag_use_annotregions == True):
        list_polygons = patient.dict_records["list_polygons"]
        if(list_polygons != None):
            for polygon in list_polygons:
                polygon = np.array(polygon) * scale_1*scale_2
                plt_polygon = patches.Polygon(polygon, facecolor='none', fill=False,\
                                              edgecolor="r", linestyle="-", linewidth=4)
                ax.add_patch(plt_polygon)
    plt.title("patient {} (extracted big/small chunks) \n file:{}"\
              .format(patient.int_uniqueid,
                      patient.dict_records["wsi"].relativedir), fontsize=20)
    plt.savefig("Visualization/LightDL/Training/patient_{}.eps"\
                .format(patient.int_uniqueid), bbox_inches='tight',  format='eps')
    plt.close(fig)
        



def otsu_getpoint_from_foreground_in_polyg(fname_wsi, num_returned_points, patient):
    #settings =======
    scale_thumbnail =  patient.dict_records["scale_thumbnail"]
    np_inpoly = (patient.dict_records["precomputed_polyongmask"][:,:,0]>0.0)
    np_otsu = patient.dict_records["precomputed_otsu"]
    w = min(np_inpoly.shape[1], np_otsu.shape[1])
    h = min(np_inpoly.shape[0], np_otsu.shape[0])
    foreground = np_inpoly[0:h, 0:w] * np_otsu[0:h, 0:w]
    #select a random point =========================
    one_indices = np.where(foreground==1.0)
    i_oneindices, j_oneindices = one_indices[0].tolist(), one_indices[1].tolist()
    n = random.choices(range(len(i_oneindices)), k=num_returned_points)
    i_oneindices, j_oneindices = np.array(i_oneindices), np.array(j_oneindices)
    i_selected, j_selected = i_oneindices[n], j_oneindices[n]
    i_selected, j_selected = np.array(i_selected), np.array(j_selected)
    #     assert(foreground[i_selected, j_selected] == 1)
    i_selected_realscale, j_selected_realscale =\
        (i_selected/scale_thumbnail).astype(np.int), (j_selected/scale_thumbnail).astype(np.int)
    x, y = list(j_selected_realscale), list(i_selected_realscale)
    return x,y 
    
class WSIRandomBigchunkLoader(BigChunkLoader):
    @abstractmethod
    def extract_bigchunk(self, last_message_fromroot):
        '''
        Extract and return a bigchunk. 
        Please note that in this function you have access to
        self.patient and self.const_global_info.
        '''
        list_bigchunks = []
        num_bigpatches = 5
        
        #preselect `num_bigpatches` random points on foreground.
        wsi = self.patient.dict_records["WSI_Her2"]
        fname_wsi = os.path.join(wsi.rootdir, wsi.relativedir)
        all_randx, all_randy = \
            otsu_getpoint_from_foreground_in_polyg(fname_wsi, num_bigpatches, self.patient)
        
        for idx_bigpatch in range(num_bigpatches): #TODO:make tunable
            #settings ==== 
            flag_use_otsu = True
            #===
            wsi = self.patient.dict_records["WSI_Her2"]
            fname_wsi = os.path.join(wsi.rootdir, wsi.relativedir)
            osimage = openslide.OpenSlide(fname_wsi)
            w, h = self.const_global_info["width_bigchunk"],\
                   self.const_global_info["heigth_bigchunk"] 
            W, H = osimage.dimensions
            rand_x, rand_y = all_randx[idx_bigpatch],\
                             all_randy[idx_bigpatch]
            rand_x, rand_y = int(rand_x-(w*0.5)), int(rand_y-(h*0.5))
            
            pil_bigchunk = osimage.read_region([rand_x, rand_y], 1, [w,h])
            np_bigchunk = np.array(pil_bigchunk)[:,:,0:3]
            patient_without_foregroundmask = copy.deepcopy(self.patient)
            for k in patient_without_foregroundmask.dict_records.keys():
                if(k.startswith("precomputed")):
                    patient_without_foregroundmask.dict_records[k] = None
            bigchunk = BigChunk(data=np_bigchunk,\
                                 dict_info_of_bigchunk={"x":rand_x, "y":rand_y},\
                                 patient=patient_without_foregroundmask)
            #log to logfile
            # ~ self.log("new bigchunk with [left,top] = [{} , {}]\n".\
                     # ~ format(rand_x, rand_y))
            list_bigchunks.append(bigchunk)
        return list_bigchunks

class WSIRandomSmallchunkCollector(SmallChunkCollector):
    def __init__(self, *args, **kwargs):
        '''
        Inputs: 
            - mode_trainortest (in const_global_info): a strings in {"train" and "test"}.
                We need this mode because, e.g., colorjitter is different in training and testing phase.
        '''
        self.mode_trainortest = kwargs["const_global_info"]["mode_trainortest"]
        assert(self.mode_trainortest in ["train", "test"])
        #grab privates
        if(self.mode_trainortest == "train"):
            self.tfms_onsmallchunkcollection =\
                torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.ColorJitter(brightness=0,\
                                         contrast=0,\
                                         saturation=0.5,\
                                         hue=[-0.1, 0.1]),\
                torchvision.transforms.ToTensor(),\
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])
            ])
        elif(self.mode_trainortest == "test"):
            self.tfms_onsmallchunkcollection =\
                torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.ToTensor(),\
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])
            ])
        super(WSIRandomSmallchunkCollector, self).__init__(*args, **kwargs)
    
    @abstractmethod     
    def extract_smallchunk(self, call_count, list_bigchunks, last_message_fromroot):
        '''
        Extract and return a smallchunk. Please note that in this function you have access to 
        self.bigchunk, self.patient, self.const_global_info.
        Inputs:
            - list_bigchunks: the list of extracted bigchunks.
        '''"list_polygons"
        if(self.mode_trainortest == "test"):
            if(call_count > 100):
                return None
        bigchunk = random.choice(list_bigchunks)
        W, H = bigchunk.data.shape[1], bigchunk.data.shape[0]
        w, h = self.const_global_info["width_smallchunk"],\
               self.const_global_info["heigth_smallchunk"]
        rand_x, rand_y = np.random.randint(0, W-w), np.random.randint(0, H-h)
        np_smallchunk = bigchunk.data[rand_y:rand_y+h, rand_x:rand_x+w, :]
        #apply the transformation ===========
        if(self.tfms_onsmallchunkcollection != None):
            toret = self.tfms_onsmallchunkcollection(np_smallchunk)
            toret = toret.cpu().detach().numpy() #[3 x 224 x 224]
            toret = np.transpose(toret, [1,2,0]) #[224 x 224 x 3]
        else:
            toret = np_smallchunk
        #wrap in SmallChunk
        smallchunk = SmallChunk(data=np_smallchunk,\
                                dict_info_of_smallchunk={"x":rand_x, "y":rand_y},\
                                dict_info_of_bigchunk = bigchunk.dict_info_of_bigchunk,\
                                patient=bigchunk.patient)
        return smallchunk



class PipelineWarwickHER2(BoVWPipeline):
    def __init__(self, num_classes, num_visualwords, device_stg3, *args, **kwargs):
        #grab privates  ====
        self.num_classes = num_classes
        self.num_visualwords = num_visualwords
        self.device_stg3 = device_stg3
        #call on super
        super(PipelineWarwickHER2, self).__init__(*args, **kwargs)
        
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



class BoVWStatCollector(StatCollector):
    def __init__(self, module_pipeline, device, *args, **kwargs):
        #grab privates
        self.module_pipeline = module_pipeline
        self.device = device
        #make other initial operations
        self.module_pipeline.to(device)
        self.module_pipeline.eval()
        self.num_calls_to_getflagfinished = 0
        super(BoVWStatCollector, self).__init__(*args, **kwargs)
        
    @abstractmethod
    def get_statistics(self, retval_collatefunc):
        x, list_patients, list_smallchunks = retval_collatefunc
        with torch.no_grad():
            netout = self.module_pipeline.testingtime_forward(x.to(self.device))#[32x100x7x7]
            netout = netout.cpu().numpy() #[32x100x7x7]
        list_statistics = []
        list_statistics += [Statistic(
                                stat=netout[n,:,:,:],\
                                source_smallchunk = list_smallchunks[n]
                              )
                             for n in range(netout.shape[0])]
        return list_statistics
    
    
    @abstractmethod
    def accum_statistics(self, prev_accum, new_stat, patient):
        if(prev_accum == None):
            #the first stat ====
            toret = {"count":1, "sum_encoded_descriptors": new_stat.stat}
        else:
            #not the first stat ====
            old_count, old_sum = prev_accum["count"], prev_accum["sum_encoded_descriptors"]
            MAXNUM_STATS = 1000 #500
            MAXNUM_STATS = 200 
            if(old_count < MAXNUM_STATS):
                toret = {"count": old_count+1,\
                         "sum_encoded_descriptors": old_sum + new_stat.stat}
            else:
                toret = {"count": old_count,\
                         "sum_encoded_descriptors": old_sum}
        return toret
                                 
    
    @abstractmethod
    def get_flag_finishcollecting(self):
        self.num_calls_to_getflagfinished += 1
        print("self.num_calls_to_getflagfinished = {}\n"\
                  .format(self.num_calls_to_getflagfinished))
        list_statcount = []
        for patient in self.dict_patient_to_accumstat.keys():
            if(self.dict_patient_to_accumstat[patient] != None):
                list_statcount.append(self.dict_patient_to_accumstat[patient]["count"])
            else:
                list_statcount.append(0)
        print(" numstats in [{} , {}],     num zeros = {}"
              .format(min(list_statcount) , max(list_statcount),
                      np.sum(np.array(list_statcount) == 0)) )
        #show bar of num-stats ====
        #plt.figure()
        #plt.bar(range(len(list_statcount)), list_statcount)
        #plt.xticks(range(len(list_statcount)), rotation='vertical')
        #plt.xlabel("index of patient")
        #plt.ylabel("number of collected stats.")
        #plt.show()
        MAXNUM_STATS = 1000 #500
        MAXNUM_STATS = 200 
        if((min(list_statcount)==MAXNUM_STATS) and (max(list_statcount)==MAXNUM_STATS)):
            if(True):#self.num_calls_to_getflagfinished > 100):
                return True
        else:
            return False

