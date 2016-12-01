'''
Alex Seewald 2016
aseewald@indiana.edu
'''
import pickle
import os
import random
import gzip
import pandas as pd
import getpass
import math
import socket
import sqlite3
import numpy as np
from scipy.stats import entropy
from enum import Enum
import datasets
import constants
import datasets

def chisquared(x,y):
    if type(x) == list or type(y) == list:
        x, y = np.array(x), np.array(y)
    return 0.5 * np.sum(np.nan_to_num(np.divide(np.square(x - y),(x+y))))

def strs_of_vars(variables,local_env):
    '''
    Takes list of variables and produces string representations of them.
    This requires searching through the environment because python does not have an easy inverse to "eval".
    Pass in locals() as local_env.
    '''
    strs = []
    for var in variables:
        for k, v in list(local_env.items()):
            if v is var:
                strs.append(k)
    return strs

class Params:
    '''
    Iterations of this project share these in common.
    If there are multiple candidate methods attempted with a shared experimentName, they end up in the same database.
    The 'type' attribute of the tables distinguishes them.
    '''
    def __init__(self,datadir,experimentName,candidate_method,objectness_method):
        self.datadir = datadir
        self.experimentName = experimentName
        if experimentName == "VOC2008":
            self.maskBased = True
            self.possible_splits = datasets.voc2008
        elif experimentName in ["COCO"]:
            self.maskBased = False
            self.possible_splits = datasets.coco
        self.candidate_method = candidate_method
        self.objectness_method = objectness_method
        self.anaconda2 = "/usr/local/anaconda2/bin/python"
        if "sqlite" in constants.dbtypes:
            self.read_db = '/data/aseewald/ctx_archu.db'
            if socket.gethostname() == 'madthunder': #the 'main' host. Others can mount related directories with sshfs.
                self.db = '/data/aseewald/ctx_archu.db'
                self.islocal = True
            else:
                self.db = self.root("ctx_arch_{}.db".format(socket.gethostname()))
                self.islocal = False
        if "postgres" in constants.dbtypes:
            # This dict will be expanded to keyword arguments in python functions making connections.
            # The settings that I use, but these can be modified without breaking these experiments.
            self.pg = {'dbname' : 'ctx2', 'user' : getpass.getuser(), 'host' : 'localhost', 'password' : 'boblawlaw'}
    def encode(self):
        '''
        This standardizes how result files will have their names indicate what experimental parameters they used.
        '''
        pass
    def root(self,arg=None):
        if arg:
            return(os.path.join(self.datadir, self.experimentName, arg))
        else:
            return(os.path.join(self.datadir, self.experimentName))
    # Some datasets like VOC2008 have ground truth available for only some of the data, so these functions return names
    # of data with and without it.
    def train_names(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("train.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("train_images"))]
    def val_names(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("val.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("val_images"))]
    def train_names_gt(self):
        "Precondition: mksplit ran."
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("train_gt.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("train_images"))]
    def val_names_gt(self,splitid=1,thresh=False):
        '''
        Precondition: mksplit ran.
        if 'tresh', another precondition is that the segments exist.
        '''
        if self.experimentName == "VOC2008":
            return [imgName[:-1] for imgName in open(self.root("val_gt.txt")).readlines()]
        else:
            return [imgName.split(".")[0] for imgName in os.listdir(self.root("val_images"))]

class GreedyParams(Params):
    '''
    See Params base class for explanations of parameters.

    og_num_scales specifies multi-scale behavoir of an older idea not considered in current implementations.
    og_k
    objgraph_distancefn is the default distance function used to define similarity of object graphs.
    '''
    def __init__(self,datadir,experimentName,splits,candidate_method,objectness_method, hs=2, og_k=4, og_num_scales=3, objgraph_distancefn=chisquared):
        Params.__init__(self,datadir,experimentName,candidate_method,objectness_method)
        self.db = '/fast-data/aseewald/ctx_arch.db'
        self.hs, self.og_k  = hs, og_k

class ArchParams(Params):
    '''
    See the command line argument help in arch.py for explanations of these variables.
    '''
    def __init__(self,datadir,experimentName,candidate_method,objectness_method,M,initialization,numfilts, \
                      conv_w,isvanilla,negprop,lr,loss_t,baseline_t,task,ctxop,include_center):
        #known baseline types.
        basetypes = ["full","biasonly","fixed_pos","patches","above-below","fixed-biasonly","attentiononly"]
        assert(baseline_t in basetypes)
        self.M,self.numfilts = M,numfilts
        self.conv_w,self.initialization,self.isvanilla = conv_w,initialization,isvanilla
        self.negprop,self.lr,self.loss_t,self.include_center = negprop,lr,loss_t,include_center
        self.task,self.ctxop,self.baseline_t  = task,ctxop,baseline_t
        Params.__init__(self,datadir,experimentName,candidate_method,objectness_method)

param_dict = {
    'DRAW4contrastive' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW4dual' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='full',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW4dual-pascal' : ArchParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='full',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW3dual' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='full',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW4-biasonly' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='biasonly',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW3-biasonly' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='biasonly',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW4-nocenter' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='full',task='discovery',ctxop='DRAW',include_center=False),
    'DRAW5-nocenter' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=5,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='full',task='discovery',ctxop='DRAW',include_center=False),
    'DRAW4-fixed' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='fixed_pos',task='discovery',ctxop='DRAW',include_center=True),
    # this is the correct "fixed" idea because constant dynamic (and non-sensicle non-trained) things are not happening.
    'DRAW4-fixedbias' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='fixed-biasonly',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW3-fixedbias' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=3,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='fixed-biasonly',task='discovery',ctxop='DRAW',include_center=True),
    'DRAW4-attentiononly' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="dual",baseline_t='attentiononly',task='discovery',ctxop='DRAW',include_center=True),
    'conv-block-intensity' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop='block_intensity',include_center=False),
    'conv-block-blur' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop='block_blur',include_center=False),
    'expanded' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=2,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='above-below',task='discovery',ctxop='expand',include_center=True),
    'above-below' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=2,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='above-below',task='discovery',ctxop='above_below',include_center=True),
    'rpatches4' : ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='patches',task='discovery',ctxop='patches',include_center=False),
    # vanilla architectures without context representation
    'vanilla_embed' : ArchParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="embed",numfilts=6,conv_w=5,isvanilla=True,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop=None,include_center=False),
    'vanilla_vgg' : ArchParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="vggnet",numfilts=6,conv_w=5,isvanilla=True,negprop=0.5,lr=1e-6,loss_t="euclidean",baseline_t='full',task='discovery',ctxop=None,include_center=False),
    'vanilla_rand' : ArchParams(datadir="/data/aseewald/",experimentName="VOC2008",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="random",numfilts=6,conv_w=5,isvanilla=True,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop=None,include_center=False)
}
params = GreedyParams("/data/aseewald/","COCO", datasets.coco,"objectness", "BING",hs=2, og_k=4)
#params = GreedyParams("/data/aseewald/","VOC2008", datasets.coco,"objectness", "BING",hs=2, og_k=4)
#params = ArchParams(datadir="/data/aseewald/",experimentName="COCO",candidate_method= "objectness",objectness_method= "BING",M=4,initialization="embed",numfilts=4,conv_w=5,isvanilla=False,negprop=0.5,lr=1e-5,loss_t="contrastive",baseline_t='full',task='discovery',ctxop='DRAW')

mainhost = "madthunder"
