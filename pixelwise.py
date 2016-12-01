'''
Alex Seewald 2016
aseewald@indiana.edu

A tensorflow implementation of Fully Convolutional Networks For Semantic Segmentation.

'''
import random
import sys
import pickle
import subprocess
import time
import itertools
import sqlite3
import os
import math
import signal
import numpy as np
import matplotlib as mpl
import multiprocessing as mp
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import deepdish as dd
import tensorflow as tf
from functools import reduce
from scipy.misc import imresize
from skimage.transform import resize
from line_profiler import *
from utils import *
import datasets
from params import params

# Making this global to keep it the same for all visualization calls.
__author__ = "Alex Seewald"

sns.set(style="whitegrid")

def hdf_restore(weights,biases,modeldir,t,sess):
    '''
    A function for resuming training having saved the parameters as a python dictionary of arrays serialized into hdf5.

    weights - a dictionary of tensorflow variables.
    biases - a dictionary of tensorflow variables.
    sess - the tensorflow session we want to restore into.
    t - the timestep within training which we restore from.
    '''
    fname = modeldir + "/" + str(t) + ".hdf"
    if not os.path.exists(fname):
        return False
    npy_w,npy_b = dd.io.load(fname)
    sess.run(tf.initialize_all_variables())
    for k in weights.keys():
        sess.run(weights[k].assign(npy_w[k]) )
    for k in biases.keys():
        sess.run(biases[k].assign(npy_b[k]))
    return True

def confusion(nickname,tstep,enforce_square=True):
    '''
    Visualization of confusion matrix.
    '''
    # put an index on 't' because it is high-cardinality and speeds this up.
    X = readsql("SELECT realcat,predcat,prob FROM fullyconv_cls WHERE nickname = '{}' AND t = {}".format(nickname,tstep))
    epsilon = X['prob'].min() / 1e4 # eliminate possibility of all-zero rows (which can't be normalized) 
    X = pd.pivot_table(X,index='realcat',columns='predcat',values='prob',fill_value=epsilon) # make like a table for heatmap, replacing nan with zero.
    if enforce_square:
        shared = set(X.columns) & set(X.index) #intersection.
        X = X.drop(set(X.index) - shared)
        for cat in set(X.columns) - shared:
            del(X[cat])
    X = X.div(X.sum(axis=1),axis=0)
    # find predcats which are not existing, and add them with all-zeros.
    sns.heatmap(X)
    plt.yticks(rotation=0)
    plt.savefig('results/confusion/{}_{}.png'.format(nickname,tstep))
    plt.show()
    plt.close()

def accuracy():
    X = readsql("SELECT * from fullyconv")
    Y = readsql("SELECT * from fullyconv")
    X['accuracy'] = X['posaccuracy']
    X['foreground?'] = True
    Y['accuracy'] = Y['negaccuracy']
    Y['foreground?'] = False
    Z = pd.concat([X,Y])
    del(Z['posaccuracy'])
    del(Z['negaccuracy'])
    g = sns.FacetGrid(Z,col="nickname",hue="foreground?",legend_out=False)
    g.map(sns.pointplot,"t","accuracy",scale=0.5)
    g.fig.get_axes()[0].legend(loc='upper left')
    plt.savefig('accuracyfc.png')
    plt.show()
    
def allvis( ):
    # visualizing confusion matrices.
    if not os.path.exists('results/confusion'):
        subprocess.call(['mkdir','results/confusion'])
    fname = 'results/fcdistinct'
    if not os.path.exists(fname):
        whichc = readsql("SELECT distinct(nickname,t) FROM fullyconv_cls")
        whichc.to_pickle(fname)
    else:
        whichc = pickle.load(open(fname,'rb'))
    for _,row in whichc.iterrows(): # the distinct function ended up storing this as text, so i have to parse to unpack.
        nickname,t = whichc.iloc[0]['row'][1:-1].split(',')
        confusion(nickname,int(t))
    accuracy()

def modeldir_of(nickname,trial,splitid,numfuse,dataset):
    '''
    Encode the hyperparameters of the training into the location in the filesystem where we store the parameters.
    '''
    modeldir = params.root('cnn/fullyconv_{}_{}_{}_{}_{}'.format(nickname,trial,splitid,numfuse,dataset))
    if not os.path.exists(modeldir):
        subprocess.call(["mkdir",modeldir])
    logdir = params.root('cnn/logsfullyconv_{}_{}_{}_{}_{}'.format(nickname,trial,splitid,numfuse,dataset))
    if not os.path.exists(logdir):
        subprocess.call(["mkdir",logdir])
    return modeldir,logdir

def signal_handler(signal,frame):
    print("Exiting prematurely, remember to manually load tsv data, like the confusion data")
    sys.exit(0)

def nearest_round(vals,domain):
    for i in range(vals.shape[0]):
        vals[i] = domain[np.argmin(np.abs(domain - vals[i]))]
        #for j in range(vals.shape[1]):
            #vals[i,j] = domain[np.argmin(np.abs(domain - vals[i,j]))]
    return vals
    
def read_dense(q,allnames,batchsize,num_classes,split,splitid=None,all_data_avail=False,dataset="COCO",anticipate_missing=False,qmax=20,saving=True,loading=True,cache_t="postproc",threaded=True,chatty=False):
    '''
    This is intended to be a function run by a multiprocessing worker. It runs indefinitely as a producer of tuples of form:
        (imgout,outgt,prop_gt_bg,names)
        where imgout is the image, outgt is the pixelwise ground truth, prop_gt is the proportion of pixels whose class label is not None.
   
    Reading this function, you may notice that pixel ground truth are stored individually per pixel in a relational database, which may seem wasteful. There is indeed a sacrifice of space and I/O time for doing things a simple way, 
    but I can spend a few hundred gigabytes and indexes on the tables make things pretty fast anyway.
 
    q - a queue to put the data once we've got it.
    qmax - if over this number of tuples in the queue, stop inserting new stuff and wait.
    allnames - list of imagenames to randomly sample from.
    batchsize - number of data to ultimately produce.
    split - pandas dataframe expressing known and unknown classes. 

    intcat in (0,len(split['known'])) are real classes.
    intcat = len(split['known'])+1 is the 'none' class.
    '''
    while True:
        qsize = q.qsize()
        if chatty: print("qsize={}".format(qsize))
        if qsize > qmax: # if queue is filling up, sleep a bit.
            time.sleep(5)
            if chatty:
                sys.stdout.write('z')
                sys.stdout.flush()
            continue
        try:
            if os.path.exists(misname): missing = np.squeeze(pd.read_csv(misname).values)
            else: missing = []
        except: missing = []
        if dataset == 'COCO':
            missing = [os.path.join(params.root('train_images'),decode_imgname(m)+".jpg") for m in missing]
            update = list(set(allnames) - set(missing))
            allnames = update
        else:
            allnames = list(set(allnames) - set(missing))
        names = random.sample(allnames,batchsize)
        if anticipate_missing: assert(allnames is not None)
        fg,allpix = 0,0
        if dataset == 'COCO':
            if not anticipate_missing: names = [decode_imgname(name) + ".jpg" for name in names]
            unoptimized_read = False
        else:
            unoptimized_read = True
        dirname = os.path.split(names[0])[0]
        if dataset == 'COCO': #using special imagename encoding.
            cachef = {imgname : "{},{}".format(splitid,encode_imgname(os.path.split(imgname)[1])) for imgname in names}
            isfast = {imgname : os.path.exists(params.root("postproc/") + cachef[imgname]) for imgname in names}
            fastlist = [encode_imgname(os.path.split(imgname)[1]) for imgname in names if isfast[imgname]]
            whichslow = [imgname for imgname in names if not isfast[imgname]]
            slowlist = ','.join(["'{}'".format(encode_imgname(os.path.split(imgname)[1])) for imgname in whichslow])
        else:
            cachef = {imgname : "{},{}".format(splitid,os.path.split(imgname)[1]) for imgname in names}
            isfast = {imgname : os.path.exists(params.root("postproc/") + cachef[imgname]) for imgname in names}
            fastlist = [os.path.split(imgname)[1] for imgname in names if isfast[imgname]]
            whichslow = [imgname for imgname in names if not isfast[imgname]]
            slowlist = ','.join(["'{}'".format(os.path.split(imgname)[1]) for imgname in whichslow])
        if chatty: print("prop(fastlist)={}".format(len(fastlist) / len(names)))
        query = "SELECT category FROM splitcats WHERE splitid = {} AND dataset = '{}' AND seen = 1".format(splitid,dataset)
        catdf = np.squeeze(readsql(query,whichdb="postgres").values)
        assert(len(catdf) > 0), "The category select query, {}, is probably wrong.".format(query)
        catlist = ','.join(["'{}'".format(category) for category in catdf])
        relname = "pixgt" if dataset == "COCO" else "pascal_pixgt"
        t0 = time.time()
        data = None
        if len(whichslow) > 0: #if whichslow is zero, all are in fastlist
            if chatty: print("Reading from database and writing to the cache")
            data = readsql("SELECT * FROM {0} R WHERE imgname IN ({1}) AND category IN ({2})".format(relname,slowlist,catlist),whichdb="postgres")
        ta = time.time()
        mymissing = []
        if anticipate_missing and len(whichslow) > 0: #if whichslow is zero, all are in fastlist
            mt0 = time.time()
            existing_names = np.unique(data['imgname'])
            names = fastlist + existing_names.tolist() #names gets re-assigned according to what is there.
            numres = existing_names.size
            ishit = [] + numres * [1] + ((batchsize - len(fastlist)) - numres) * [0]
            numslow = np.unique(data['imgname']).size
            while numslow < (batchsize - len(fastlist)):
                difference = (batchsize - len(fastlist)) - numslow #the number we want to get.
                if dataset == 'COCO':
                    fillin_names = [encode_imgname(os.path.split(x)[1]) for x in random.sample(allnames,round(1.9 * difference))]
                else:
                    fillin_names = random.sample(allnames,round(2.1 * difference))
                fillname_list = ','.join(["'{}'".format(fname) for fname in fillin_names])
                newdata = readsql("SELECT * FROM {0} WHERE imgname IN ({1}) AND category IN ({2})".format(relname,fillname_list,catlist),whichdb="postgres")
                unique_new = list(newdata['imgname'].unique())
                if len(unique_new) > difference:
                    if chatty: print("Overshot it by {}".format(len(unique_new) - difference))
                    unique_new = unique_new[0:difference]
                    newdata = newdata[newdata['imgname'].isin(unique_new)]
                data = data.append(newdata)
                for fname in fillin_names:
                    hit = fname in unique_new
                    ishit.append(hit)
                    if hit:
                        numslow += 1
                        sys.stdout.write('*')
                        names.append(fname)
                    else:
                        mymissing.append(fname)
                        sys.stdout.write('.')
                    sys.stdout.flush()
                    if numslow == (batchsize - len(fastlist)):
                        break
            sys.stdout.write('\n')
            if chatty: print("proportion of times imgname is in the database: {}, time spent adding missing={}".format(np.mean(ishit),time.time() - mt0))
        assert(len(names) == batchsize), "loop adding names exited too soon"
        try:
            imgs = OrderedDict({name : (imread_wrap(name,tmp=unoptimized_read),imread(name)) for name in names})
        except:
            suffix = ".jpg" if "jpg" not in names[0] else ""
            if dataset == "COCO":
                try: #this occasionally fails for some reason.
                    imgs = OrderedDict([(name, (imread_wrap(os.path.join(dirname,decode_imgname(name + suffix)),tmp=unoptimized_read),imread(os.path.join(dirname,decode_imgname(name + suffix))))) for name in names])
                except:
                    print("Failed to imread,continuing")
                    continue
            else:
                imgs = OrderedDict([(name,(imread_wrap(os.path.join(dirname,name),tmp=unoptimized_read),imread(os.path.join(dirname,name)))) for name in names])
        # keep a record of missing data, so I can not waste time querying for it. If training and adding data at the same time, we will want to periodically delete this file.
        with open(misname,'a') as mif:
            for mis in mymissing:
                mif.write("{}\n".format(mis))
        mymissing = []
        t1 = time.time()
        sizes = {name : img[1].shape for (name,img) in imgs.items()}
        tmp = OrderedDict([(name,num_classes * np.ones((sizes[name][0],sizes[name][1]))) for name in set(names) - set(fastlist)])
        gt = num_classes * np.ones((batchsize,224,224,num_classes+1))
        unknownError = False
        if data is not None: #'data' will be none if its all in the fastlist.
            for imgname,df in data.groupby('imgname'):
                if dataset == 'COCO':
                    k = encode_imgname(imgname)
                else:
                    k = os.path.join(dirname,imgname)
                if k not in set(names) - set(fastlist):
                    continue
                for category,dfp in df.groupby('category'):
                    if category not in np.squeeze(split['category'].values):
                        continue
                    if splitid != None:
                        intcat = split[split['category'] == category].index[0]
                    try: #how did this get so complicated?
                        tmp[k][dfp[['y','x']].values.T.tolist()] = intcat
                    except:
                        print("k={},names={},k in names = {}".format(k,names,k in names))
                        try:
                            tmp[os.path.split(k)[1]][dfp[['y','x']].values.T.tolist()] = intcat
                        except:
                            unknownError = True
                            break
                if unknownError: break
            if unknownError:
                print("Weird error with tmp keys")
                continue
        try:
            i = 0
            ordered_names = []
            for k in fastlist:
                sys.stdout.write('~')
                g = onehot(pickle.load(open(params.root('postproc/' + splitid + "," + k),'rb')),num_classes+1)
                gt[i] = g
                i += 1
                ordered_names.append(k)
            for k in tmp.keys():
                gtshaped = resize(tmp[k],(224,224),order=0)
                mask = np.add.reduce(np.array([gtshaped == val for val in np.unique(tmp[k])]),0).astype(np.bool)
                gtshaped[~mask] = nearest_round(gtshaped[~mask],np.unique(tmp[k]))
                fg += np.count_nonzero(gtshaped - num_classes)
                allpix += gtshaped.size
                try:
                    if not os.path.exists(params.root('postproc/' + splitid + "," + k)):
                        pickle.dump(gtshaped,open(params.root('postproc/' + splitid + "," + k),'wb'))
                except:
                    print("failed to write to cache, continuing")
                if i < batchsize: #otherwise, we just dump them out.
                    gt[i] = onehot(gtshaped,num_classes+1)
                    i += 1
                    ordered_names.append(k)
            imgout = np.array([imgs[k][0] for k in ordered_names])
            try:
                assert(imgout.shape[0] == batchsize), "imgout.shape={},batchsize={}".format(imgout.shape,batchsize)
                # multiplying by num_classes to get proportion foreground because there is one-hot encoding.
                prop_gt_bg = 1.0 - fg/allpix
                outgt = gt.reshape(gt.shape[0],gt.shape[1] * gt.shape[2],gt.shape[3])
                if (outgt.shape[0] > batchsize) or imgout.shape[0] > batchsize: #why would this possible happen? seems it did, so just handle it.
                    outgt = outgt[0:batchsize]
                    imgout = imgout[0:batchsize]
                if chatty: print("Proportion background: {}, threadid={} took {} seconds to load db, prop time on query={}".format(prop_gt_bg,mp.current_process(),t1 - t0,(ta - t0)/(time.time() - t0)))
                assert(imgout.shape == (batchsize,224,224,3) and outgt.shape == (batchsize,224*224,num_classes+1)), "imgout.shape={} and outgt.shape={}".format(imgout.shape,outgt.shape,len(names))
            except:
                print("Bad shape, going to next iteration")
                continue
        except:
            print("some problem with gathering gt and getting it in right shape.")
            continue
        out = (imgout,outgt,prop_gt_bg,names)
        if threaded:
            q.put(out)
        else:
            return out

def outer_vis(dataset,split,num_classes,splitid):
    '''
    A closure-returning function for visualizing things. We use a closure to keep bound variables related to colors consistent.
    '''
    classnames = list(np.squeeze(split['category'].values)) + ['None']
    fname = 'cache/{}_colors_{}'.format(dataset,splitid)
    if not os.path.exists(fname): #make consistent colors for all figures by pickling it.
        colors = [(1,1,1)] + [(random.random(),random.random(),random.random()) for i in range(len(classnames))]
        randmap = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=len(classnames))
        pickle.dump( (colors,randmap), open(fname,'wb'))
    else:
        colors,randmap = pickle.load(open(fname,'rb'))
    def visualize_net(out,img):
        fig,axes = plt.subplots(2)
        plt.gcf().set_size_inches(36,36)
        axes[0].imshow(img)
        axout = axes[1].matshow(out,cmap=randmap)
        for ax in axes: ax.grid('off')
        formatter = plt.FuncFormatter(lambda val,loc: classnames[val])
        plt.colorbar(axout,ticks=range(len(classnames)),format=formatter)
    def visualize_compare(out,numfuse,img):
        fig,axes = plt.subplots(numfuse+3)
        fig.set_size_inches(36,36)
        ax = 0
        for k,v in out.items():
            axes[ax].matshow(out[k],cmap=randmap)
            axes[ax].set_title(k)
            ax += 1
        axes[-1].imshow(img)
        axes[-1].set_title("image")
        for ax in axes: ax.grid('off')
    def visualize(rawimg,imgin,imgout,imgname,t,splitid,numfuse,title):
        if imgout.shape[-1] == (num_classes+1): #haven't reduced to argmax yet.
            confidence,ncol = True,4
        else:
            confidence,ncol = False,3
        fig,axes = plt.subplots(ncols=ncol)
        fig.set_size_inches(30,10)
        fig.suptitle(title)
        axes[0].imshow(rawimg)
        axin = axes[1].matshow(imgin,cmap=randmap,vmin=0,vmax=len(classnames))
        if confidence:
            axout = axes[2].matshow(np.max(imgout,axis=2),cmap=randmap,vmin=0,vmax=len(classnames))
            axes[3].matshow(np.max(imgout,axis=2) / np.sum(imgout,axis=2))
        else:
            axout = axes[2].matshow(imgout,cmap=randmap,vmin=0,vmax=len(classnames))
        formatter = plt.FuncFormatter(lambda val,loc: classnames[val])
        fig.colorbar(axout,ticks=range(len(classnames)),format=formatter)
        if not os.path.exists(params.root(params.root('results/fullyconv'))):
            subprocess.call(["mkdir","-p",params.root(params.root('results/fullyconv'))])
        plt.savefig(params.root('results/fullyconv/{}_{}_{}_{}.png'.format(t,os.path.split(imgname)[1],splitid,numfuse)))
        plt.close()
    return visualize,visualize_net,visualize_compare

def update_bgscale(prop_pred_bg,prop_gt_bg,bgscale,adjust_rate=0.02):
    '''
    When treating all classes equally in the loss function, the model is biased towards classifying pixels as background.
    The number returned by this function is the weight (should be less than 1) 

    prop_pred_bg - proportion of pixels predicted as background.
    prop_gt_bg - proportion of pixels labeled as background in ground truth.
    bgscale - current bgscale, to be updated.
    '''
    # when this term gets very large things blow up. That happened accidentally, but might as well put "min" in there to avoid anything horrible.
    return min(math.exp(adjust_rate * (prop_gt_bg - prop_pred_bg)) * bgscale,0.09)

def create_tables( ):
    '''
    Keep track of training statistics in relational databases.
    '''
    dosql("CREATE TABLE IF NOT EXISTS fullyconv(nickname TEXT, trial INT, t INT,name TEXT,walltime DATE,loss_amount FLOAT,samples INT,posaccuracy FLOAT,negaccuracy FLOAT,numfuse INT)",whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS fullyconv_cls(nickname TEXT, trial INT, t INT,name TEXT,realcat TEXT,predcat TEXT, prob FLOAT, numfuse INT)",whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS splitcats(splitid INT,seen INT, category TEXT)",whichdb="postgres")
    dosql("CREATE TABLE IF NOT EXISTS fuseconst(nickname TEXT, t INT, layer TEXT, const FLOAT)")

def setup(sess,trestart,nickname,numfuse,use_bias,dataset,split,splitid,batchsize,threaded,all_data_avail,anticipate_missing,placeholders,num_readers=1,starting=None,train=True):
    '''
    Setup that needs to be done by train and test, so abstract it as a function.
    '''
    _X,_pix,_dropout,_bgscale = placeholders
    num_classes = len(split)
    if dataset == 'COCO':
        traindir,valdir = params.root('train_images'),params.root('val_images')
        const_imgnames = [ ] # need to pick out some for illustration's sake.
    else:
        traindir,valdir = '/data_b/aseewald/data/VOC2008/JPEGImages','/data_b/aseewald/data/VOC2008/JPEGImages'
        const_imgnames = [ ]
    # determine which 'trial' is happening.
    trial = readsql("SELECT max(trial) FROM fullyconv WHERE nickname = '{}'".format(nickname))
    if starting == "newmax":
        print("Doing new trial.")
        if trial is None or len(trial) == 0 or trial.values[0][0] == None: trial = 0
        else: trial = np.squeeze(trial.values)[0] + 1
    elif starting is None:
        print("Using parameters from previous max")
        if trial is None or len(trial) == 0 or trial.values[0][0] == None: trial = 0
        else: trial = trial.values[0][0]
    else:
        assert(isinstance(starting,int))
        print("Starting is an int, so restarting at {}th trial".format(starting))
        trial = starting
    if all_data_avail:
        train_names = os.listdir(traindir)
        val_names = os.listdir(valdir)
        print("Training with all data.")
    else:
        # CREATE MATERIALIZED VIEW 
        relname = "coco" if dataset == "COCO" else "pascal"
        if anticipate_missing: #we will guess and check.
            if dataset == 'COCO':
                train_names = os.listdir(params.root('train_images'))
                val_names = os.listdir(params.root('val_images'))
            else:
                raise NotImplementedError
            assert(not all_data_avail) #there's no reason for that to ever be the case.
        else: # we have precomputed the view of present images.
            train_names = np.squeeze(readsql("SELECT imgname FROM {}".format(trainview),whichdb="postgres").values).tolist()
        print("Training with {} of the data".format(len(train_names) / len(os.listdir(traindir))))
    if dataset == 'pascal':
        val_names = train_names
        num_epochs = 50 #do more training epochs because there is less data.
    else:
        val_names = [os.path.join(valdir,name) for name in val_names]
        num_epochs = 4
    # make them absolute paths.
    train_names = [os.path.join(traindir,name) for name in train_names]
    scales = {}
    if numfuse == 0:
        scales['upsample5'] = 1.0
    else:
        scale_data = readsql("SELECT * FROM fuseconst WHERE nickname = '{}' AND t = {}".format(nickname,trestart))
        if len(scale_data) == 0:
            if numfuse == 1: #needs to be normalized, so one degree of freedom here.
                scales['upsample5'] = tf.Variable(0.5,dtype=tf.float32,name="scale5",trainable=True)
            elif numfuse == 2:
                scales['upsample5'] = tf.Variable(0.3333,dtype=tf.float32,name="scale5",trainable=True)
                scales['upsample4'] = tf.Variable(0.3333,dtype=tf.float32,name="scale4",trainable=True)
            else:
                return False
            print("No scale data saved, so starting with equal weights")
        else:
            for sd in scale_data.iterrows():
                scales[sd['layer']] = sd['const']
            assert(len(scales.keys()) == numfuse),"not all the scale data saved, delete incomplete data from fuseconst and start with uniform data"
    modeldir,logdir = modeldir_of(nickname,trial,splitid,numfuse,dataset)
    queue = mp.Queue()
    if threaded:
        for i in range(num_readers):
            if train:
                args = (queue,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing)
            else:
                args = (queue,val_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing)
            proc = mp.Process(target=read_dense,args=args)
            proc.start()
            print("Started reader",i)
        print("Started all the readers")
    print("Before the get")
    parameters = initialize(num_classes,numfuse)
    if threaded:
        dfst = queue.get()
    else:
        dfst = read_dense(queue,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,threaded=False)
    print("Did a get successfully")
    debug_info = {'feed' : {_X : dfst[0],_pix : dfst[1],_bgscale : 0.05, _dropout : 1.0},'sess' : sess}
    saver = tf.train.Saver(max_to_keep=50)
    isaccurate,loss,optimizer,outimgs,inimgs,outdict,lossfg,lossreg = mkopt(_X,_pix,parameters,_dropout,num_classes,batchsize,numfuse,_bgscale,scales,di=debug_info,use_bias=use_bias)
    sess.run(tf.initialize_all_variables()) # run it again, now that AdamOptimizer created some new variables. No evidence there is a problem with doing it twice.
    if os.path.exists(params.root(modeldir)) and len(os.listdir(params.root(modeldir))) > 0:
        assert(trestart is not None), "if hdf, you must provide trestart"
        hdf_restore(parameters[0],parameters[1],modeldir,trestart,sess)
        print("Sucessfully restored from iteration",trestart)
    else:
        print("Starting from scratch")
        trestart = 0
    return queue,parameters,isaccurate,loss,optimizer,outimgs,inimgs,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart

def train(nickname,numfuse=1,splitid=None,all_data_avail=False,do_refresh=False,dataset="COCO",bgscale=0.05,starting=None,anticipate_missing=False,device="GPU",savestep=100,use_bias=1,threaded=True,trestart=None,savemethod="hdf",batchsize=46,valstep=40,visstep=20,infreq_visstep=60,biasstep=40,num_test_batches=4):
    
    '''
    numfuse - number of convolutional layers to upsample and add together to get a "fused" pixelwise prediction.
    do_refresh - if True, refresh materialized views of relevant imgnames.
    timeout - number of seconds after which to interrupt training. Set by default to 36 hours.
    all_data_avail - if True, don't check whether data exists. 
    do_refresh - set of available image names are cached because full tables are many Gigs. Refreshing the cache takes awhile.
    num_readers - The database reads happen in another process communicating with this function over a pipe. this argument is the number of such processes.
    val_step - Run testing of performance with this period.
    vis_step - Visualize classifications with this period.
    infreq_visstep - Additional visualizations with this period.
    biasstep - Analyze biases of the network with this period.

    Other arguments are assumed self-explanatory or equal in name to something explained elsewhere.
    '''
    global misname
    signal.signal(signal.SIGINT,signal_handler)
    misname = 'cache/missing_{}_{}.pkl'.format(nickname,splitid)
    predcounts = []
    if dataset == 'pascal':
        anticipate_missing = False
    walltime_0 = time.time()
    create_tables()
    cls_tsv = open('fc-cls-cache_{}.tsv'.format(nickname),'a')
    dosql("CREATE TABLE IF NOT EXISTS fullyconv_settings(nickname TEXT, pkl TEXT)")
    setname = params.root('settings/fullyconv_{}'.format(nickname))
    dosql("INSERT INTO fullyconv_settings VALUES ('{}','{}')".format(nickname,setname))
    if not os.path.exists(params.root('settings')):
        subprocess.call(["mkdir",params.root('settings')])
    if not os.path.exists(setname):
        pickle.dump({'numfuse' : numfuse, 'usebias' : use_bias},open(setname,'wb'))
    bg_hist = []
    if len(readsql("SELECT * FROM splitcats",whichdb="postgres")) == 0:
        addcat(split)
    split = readsql("SELECT * FROM splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 1".format(dataset,splitid),whichdb="postgres")
    num_classes = len(split)
    visualize = outer_vis(dataset,split,num_classes,splitid=splitid)
    sessname = str(splitid)
    cats = ['' for cat in split['category']]
    for category in split['category'].values:
        intcat = split[split['category'] == category].index[0]
        cats[intcat] = category
    cats.append('None')
    assert(num_classes+1 == len(cats))
    if not os.path.exists(params.root('gtprop')):
        subprocess.call(["mkdir",params.root('gtprop')])
    gtpname = params.root('gtprop/{}.pkl'.format(splitid))
    if not os.path.exists(gtpname):
        gtprop = []
    else:
        gtprop = pickle.load(open(gtpname,'rb'))
    if not os.path.exists('cache/verify_split_order_{}'.format(nickname)):
        pickle.dump(cats,open('cache/verify_split_order_{}'.format(nickname),'wb'))
    else:
        assert(cats == pickle.load(open('cache/verify_split_order_{}'.format(nickname),'rb'))),"ordering is different on different runs. This should be impossible."
    pc_cols = np.concatenate((split['category'].values,['None','timestep']))
    if device == "GPU":
        devstr,checkup = '/gpu:0',True
    elif device == "CPU": #turn off checkup to save time when using CPU.
        devstr,checkup = '/cpu:0',False
    num_epochs = 50 if dataset == 'pascal' else 4 #do more training epochs because there is so little data.
    with tf.device(devstr):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
            _X = tf.placeholder(tf.float32,[batchsize,224,224,3])
            _pix = tf.placeholder(tf.float32,[batchsize,224 * 224,num_classes+1])
            _dropout = tf.placeholder(tf.float32,shape=())
            _bgscale = tf.placeholder(tf.float32,shape=())
            placeholders = (_X,_pix,_dropout,_bgscale)
            # trestart is both an argument and a return value because it is conditionally updated.
            queue,parameters,isaccurate,loss,optimizer,outimgs,inimgs,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart = setup(sess,trestart,nickname,numfuse,use_bias,dataset,split,splitid,batchsize,threaded,all_data_avail,anticipate_missing,placeholders,starting=starting,train=True)
            modeldir,logdir = modeldir_of(nickname,trial,splitid,numfuse,dataset)
            if not os.path.exists(modeldir): subprocess.call(["mkdir",modeldir])
            num_batches = int(num_epochs * len(train_names) // batchsize)
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(logdir + '/train', sess.graph)
            pos_accurate,neg_accurate = isaccurate
            print("About to begin training")
            for batchid in range(trestart,num_batches):
                epoch = (batchid * batchsize) / len(train_names)
                sys.stdout.write("datetime={},batchid={},epoch={}".format(time.asctime(),batchid,epoch))
                if threaded:
                    X,gt,prop_gt_bg,batchnames = queue.get()
                else:
                    X,gt,prop_gt_bg,batchnames = read_dense(queue,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,threaded=False)
                gtprop.append(np.mean(gt,axis=(0,1)))
                if batchid % 10 == 0:
                    pickle.dump(gtprop,open(gtpname,'wb'))
                sys.stdout.write(',got\n')
                if (batchid % visstep == 0) and checkup: # some visualizations.
                    feed = {_X : X, _pix : gt, _dropout : 0.4, _bgscale : bgscale}
                    imgsout,imgsin,lossfgamt,lossamt,lossregamt,_ = sess.run([outimgs,inimgs,lossfg,loss,lossreg,optimizer], feed_dict=feed)
                    imgsout = np.squeeze(imgsout)
                    print("loss={},loss_fg={},loss_reg={},loss_prop_fg={}".format(lossamt,lossfgamt,lossregamt/lossamt,lossfgamt/lossamt))
                    try:
                        tn = time.time()
                        numrand = min(batchsize,2)
                        for i in random.sample(list(range(len(batchnames))),numrand):
                            visualize(X[i],imgsin[i],imgsout[i],batchnames[i],batchid,splitid,numfuse,"fused")
                        print("Frequent Visualization took {} seconds".format(time.time() - tn))
                    except: print("Something wrong with visualiation.")
                elif (batchid % infreq_visstep == 1) and checkup: # more extensive visualizations.
                    print("Starting infrequent visualization")
                    feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
                    try:
                        for k in outdict:
                            imgsout = sess.run(outdict[k],feed)
                            imgsout = np.squeeze(imgsout).reshape((batchsize,224,224,num_classes+1))
                            for i in random.sample(list(range(len(batchnames))),numrand):
                                visualize(X[i],imgsin[i],imgsout[i],batchnames[i],batchid,splitid,numfuse,"fused")
                        print("finished infrequent visualization")
                    except: print("Something wrong with infrequent visualization")
                else: #just running the optimizer.
                    feed = {_X : X, _pix : gt, _dropout : 0.3, _bgscale : bgscale}
                    t0 = time.time()
                    pred,outfull,lossamt,lossfgamt,lossregamt,_ = sess.run([outimgs,outdict['upsample5'],loss,lossfg,lossreg,optimizer], feed_dict=feed)
                    pixprobs = normalize_unscaled_logits(outfull)
                    minprob,maxprob = np.min(pixprobs),np.max(pixprobs)
                    print("lossamt={},lossfg_prop={},lossreg_prop={},minprob={},maxprob={}".format(lossamt,lossfgamt/lossamt,lossregamt/lossamt,minprob,maxprob))
                    if batchid % biasstep == 0:
                        for intcat in range(num_classes+1):
                            minc,maxc = np.min(pixprobs[:,:,:,intcat]),np.max(pixprobs[:,:,:,intcat])
                            cat = cats[intcat] if intcat < num_classes else 'None'
                            print("cat={},min={},max={}".format(cat,minc,maxc))
                            plt.title(cat + ' distribution')
                            plt.hist(pixprobs[:,:,:].flatten())
                            outn = params.root("results/{}".format(nickname))
                            if not os.path.exists(outn): subprocess.call(["mkdir",outn])
                            plt.savefig(outn + "/" + str(batchid) + "_" + cat)
                            plt.close()
                    prop_pred_bg = np.count_nonzero(pred == num_classes) / pred.size
                    histogram = np.bincount(pred.flatten())
                    predcount = np.concatenate((histogram,np.zeros(num_classes+1 - len(histogram)),[batchid]))
                    predcounts.append(predcount)
                    print("prediction frequency: ",list(zip(pc_cols,predcount)))
                    bgscale = update_bgscale(prop_pred_bg,prop_gt_bg,bgscale)
                    if bgscale > 1: print("Warning: bgscale is getting big and it is getting weird.")
                    else: print("Proportion predicted bg: {} ,New bgscale: {}".format(prop_pred_bg,bgscale))
                    bg_hist.append(bgscale)
                    print("Optimizer took {} seconds".format(time.time() - t0))
                # TEST
                if (batchid % valstep == 0) and checkup:
                    for i in range(num_test_batches):
                        if threaded:
                            X,gt,prop_gt_bg,_ = queue.get()
                        else:
                            X,gt,prop_gt_bg,_ = read_dense(queue,train_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,threaded=False)
                        feed_acc = {_X : X, _pix : gt, _dropout : 1.0, _bgscale : bgscale}
                        posaccuracy,negaccuracy,loss_amount,summary = sess.run([pos_accurate,neg_accurate,loss,merged],feed_dict=feed_acc)
                        train_writer.add_summary(summary,batchid)
                        posaccuracy,negaccuracy = np.mean(posaccuracy),np.mean(negaccuracy)
                        print("posaccuracy={},negaccuracy={}".format(posaccuracy,negaccuracy))
                        q_acc = lambda x:"INSERT INTO fullyconv VALUES('{}',{},{},'{}',{},{},{},{},{})".format(nickname,trial,batchid,sessname,x,loss_amount,len(X),posaccuracy,negaccuracy,numfuse)
                        dosql(xplatformtime(q_acc),whichdb="postgres")
                        try:
                            gtshaped = gt.reshape((batchsize,224,224,num_classes+1))
                            gtmeans = np.mean(gtprop,axis=0)
                            # so that when we take min, we don't count the zero-entries ( )
                            gtmin = np.min(gtmeans[gtmeans > 0])
                            for intcat in range(num_classes+1):
                                catname,gtc = cats[intcat],gtmeans[intcat]
                                if random.random() < 0.1: #lots of data would get written, so limit it a bit.
                                    for b in range(pred.shape[0]):
                                        for py in range(pred.shape[1]):
                                            for px in range(pred.shape[2]):
                                                # class balance the confusion matrix data.
                                                if random.random() <= (gtmin / max(gtc,gtmin)): # max to avoid division by zero when starting.
                                                    tup = (nickname,trial,batchid,cats[np.argmax(gtshaped[b,py,px])],catname,pixprobs[b,py,px,intcat],numfuse)
                                                    cls_tsv.write('\t'.join(map(str,tup)) + '\n')
                        except:
                            print("Failed to write confusion data")
                if batchid % savestep == (savestep - 1):
                    # saving hdf style
                    tsa= time.time()
                    # keys and values get saved in the same order, so this dependence on ordering works.
                    w_keys,b_keys = list(parameters[0].keys()),list(parameters[1].keys())
                    w_out,b_out = sess.run(list(parameters[0].values())), sess.run(list(parameters[1].values()))
                    weight_snapshot = OrderedDict({k : w_out[w_keys.index(k)] for k in parameters[0].keys()})
                    bias_snapshot = OrderedDict({k : b_out[b_keys.index(k)] for k in parameters[1].keys()})
                    dd.io.save(modeldir + "/" + str(batchid) + ".hdf",(weight_snapshot,bias_snapshot))
                    # saving tf style.
                    print("Saving at batchid={}".format(batchid))
                    pc = pd.DataFrame(predcounts,columns=pc_cols)
                    pc.to_pickle('cache/predcounts_{}'.format(nickname)) 
                    pickle.dump(bg_hist,open('cache/{}_bg_hist'.format(nickname),'wb'))
                    saver.save(sess,modeldir + "/" + "model",global_step=batchid)
                    print("Done saving, which took {} seconds".format(time.time() - tsa))

def initialize(num_classes,numfuse,pretrained='vggnet.npy',hack_suffix=None):
    '''
    Goal: eventually replace pretrained with earlier conv weights learned on COCO pixelwise.
    '''
    try:
        vgg = np.load(params.root('cnn/npymodels/{}'.format(pretrained)),encoding='bytes').item()
    except:
        vgg = np.load('{}'.format(pretrained),encoding='bytes').item()
    take = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3']
    weights = {}
    for layer in take:
        weights[layer] = vgg[layer]
    weights['myconv5_1'] = [0.01 * np.random.randn(3,3,512,512), 0.001 * np.random.randn(512)]
    weights['myconv5_2'] = [0.01 * np.random.randn(3,3,512,512), 0.001 * np.random.randn(512)]
    weights['upsample5'] = [0.01 * np.random.randn(224/7,224/7,num_classes+1,num_classes+1), 0.0001 * np.random.randn(num_classes+1)]
    weights['narrow5'] = [0.01 * np.random.randn(1,1,512,num_classes+1), 0.0001 * np.random.randn(num_classes+1)]
    if numfuse >= 1:
        weights['upsample4'] = [0.01 * np.random.randn(224/14,224/14,num_classes+1,num_classes+1), 0.0001 * np.random.randn(num_classes+1)]
        weights['narrow4'] = [0.01 * np.random.randn(1,1,512,num_classes+1), 0.0001 * np.random.randn(num_classes+1)]
    if numfuse >= 2:
        weights['upsample3'] = [0.01 * np.random.randn(224/28,224/28,num_classes+1,num_classes+1), 0.01 * np.random.randn(num_classes+1)]
        weights['narrow3'] = [0.01 * np.random.randn(3,3,256,num_classes+1), 0.01 * np.random.randn(num_classes+1)]
    return(totensors(weights,trainable=True,xavier={k : False for k in weights.keys()},hack_suffix=hack_suffix))

def run(nickname,trial,splitid,numfuse,imgdir,batchsize=40,topk=6):
    i = 0
    imgnames = [os.path.join(imgdir,x ) for x in os.listdir(imgdir)]
    dataset = 'COCO'
    modeldir,logdir = modeldir_of(nickname,trial,splitid,numfuse,dataset)
    split = pd.read_sql("SELECT * FROM splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 1".format(dataset,splitid),sqlite3.connect('splitcats.db'))
    num_classes = len(split)
    classnames = np.array(list(np.squeeze(split['category'].values)) + ['None'])
    outdir = 'demo-results'
    if not os.path.exists(outdir):
        subprocess.call(["mkdir",outdir])
    if not os.path.exists("cache"):
        subprocess.call(["mkdir","cache"])
    _,visualize_net,visualize_compare = outer_vis(dataset,split,num_classes,splitid=splitid)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        parameters = initialize(num_classes,numfuse)
        try:
            fs = [x for x in os.listdir(params.root(modeldir)) if "hdf" in x and "best.hdf" not in x]
            existing_ts = [int(os.path.split(x)[1].split('.')[0]) for x in fs]
            hdf_restore(parameters[0],parameters[1],modeldir,np.max(existing_ts),sess)
        except:
            try:
                hdf_restore(parameters[0],parameters[1],os.getcwd(),7279,sess)
            except:
                print("A trained model does not exist, contact Alex at aseewald@indiana.edu to determine the issue")
                sys.exit(1)
        # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
        _X = tf.placeholder(tf.float32,[batchsize,224,224,3])
        _dropout = tf.placeholder(tf.float32,shape=())
        outs = arch(_X,parameters[0],parameters[1],_dropout,num_classes,batchsize,numfuse)
        while i < len(imgnames):
            amount = min(len(imgnames)-i,batchsize)
            X = [imread_wrap(imgnames[i+j]) for j in range(amount)]
            feed = {_X : X, _dropout : 1.0}
            out = sess.run(outs,feed)
            out = {k : normalize_unscaled_logits(v) for k,v in out.items()}
            for j in range(amount):
                net = np.argmax(out['net'][j].reshape((224,224,num_classes+1)),axis=2)
                tail = os.path.split(imgnames[i+j])[1]
                visualize_net(net,X[j])
                plt.savefig(os.path.join(outdir,"net-" + tail))
                which = np.flipud(np.argsort(np.sum(out['net'][j],axis=0)))[0:topk]
                k_cat = classnames[which]
                print("Top k categories in image={} are {}".format(tail,k_cat))
            for j in range(amount):
                visualize_compare({k : np.argmax(out[k][j].reshape((224,224,num_classes+1)),axis=2) for k in out.keys()},numfuse,X[j])
                plt.savefig(os.path.join(outdir,"comparison-" + os.path.split(imgnames[i+j])[1]))
            plt.close("all")
    
def test(nickname,numfuse=1,splitid=None,all_data_avail=False,do_refresh=False,dataset="COCO",bgscale=0.05,starting=None,anticipate_missing=False,device="GPU",savestep=100,use_bias=1,threaded=False,trestart=None,savemethod="hdf",batchsize=46,valstep=40,visstep=20,infreq_visstep=60,biasstep=40,num_test_batches=4):
    misname = 'cache/missing_{}_{}.pkl'.format(nickname,splitid)
    predcounts = []
    if dataset == 'pascal':
        anticipate_missing = False
    walltime_0 = time.time()
    create_tables()
    split = readsql("SELECT * FROM splitcats WHERE dataset = '{}' AND splitid = {} AND seen = 1".format(dataset,splitid),whichdb="postgres")
    num_classes = len(split)
    visualize = outer_vis(dataset,split,num_classes,splitid=splitid)
    sessname = str(splitid)
    cats = ['' for cat in split['category']]
    for category in split['category'].values:
        intcat = split[split['category'] == category].index[0]
        cats[intcat] = category
    cats.append('None')
    assert(num_classes+1 == len(cats))
    pc_cols = np.concatenate((split['category'].values,['None','timestep']))
    if device == "GPU":
        devstr,checkup = '/gpu:0',True
    elif device == "CPU": #turn off checkup to save time when using CPU.
        devstr,checkup = '/cpu:0',False
    agg_confusion = []
    with tf.device(devstr):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
            _X = tf.placeholder(tf.float32,[batchsize,224,224,3])
            _pix = tf.placeholder(tf.float32,[batchsize,224 * 224,num_classes+1])
            _dropout = tf.placeholder(tf.float32,shape=())
            _bgscale = tf.placeholder(tf.float32,shape=())
            placeholders = (_X,_pix,_dropout,_bgscale)
            # trestart is both an argument and a return value because it is conditionally updated.
            queue,parameters,isaccurate,loss,optimizer,outimgs,inimgs,outdict,lossfg,lossreg,saver,train_names,val_names,trial,trestart = setup(sess,trestart,nickname,numfuse,use_bias,dataset,split,splitid,batchsize,threaded,all_data_avail,anticipate_missing,placeholders,starting=starting)
            modeldir,logdir = modeldir_of(nickname,trial,splitid,numfuse,dataset)
            batchid = 0
            while True:
                if threaded:
                    X,gt,prop_gt_bg,_ = queue.get()
                else:
                    X,gt,prop_gt_bg,_ = read_dense(queue,val_names,batchsize,num_classes,split,splitid,False,dataset,anticipate_missing,threaded=False)
                feed_acc = {_X : X, _pix : gt, _dropout : 1.0, _bgscale : bgscale}
                pred,posaccuracy,negaccuracy,loss_amount,summary = sess.run([outimgs,pos_accurate,neg_accurate,loss,merged],feed_dict=feed_acc)
                print("Correctly classified {} of foreground pixels".format(posaccuracy))
                print("Correctly classified {} of background pixels".format(negaccuracy))
                confusion = np.zeros(nclasses,nclasses)
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[1]):
                        for k in range(pred.shape[2]):
                            pass 
                print("Saved current confusion matrix to {}")
                batchid += 1

def arch(X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=None,di=None,use_bias=True):
    '''
    The beginning of the architecture is a VGGnet.
    X - (?,224,224,3)
    weights - a dictionary containing tensorflow variables. It is defined in the initialize function.
    biases - similar, but for biases.
    num_classes - 
    batchsize - here, treated as a constant.
    numfuse - 
    '''
    # use normalized but parameterized scales.
    scales = {}
    if numfuse == 0:
        scales['upsample5'] = 1.0
    elif numfuse == 1:
        if alphas is not None:
            scales['upsample5'] = alphas['upsample5']
            scales['upsample4'] = 1 - scales['upsample5']
        else:
            scales['upsample5'],scales['upsample4'] = 0.5,0.5
    elif numfuse == 2:
        if alphas is not None:
            scales['upsample5'] = alphas['upsample5']
            scales['upsample4'] = alphas['upsample4']
            scales['upsample3'] = 1 - (scales['upsample4'] + scales['upsample5'])
        else:
            scales['upsample5'],scales['upsample3'],scales['upsample4'] = 0.333,0.333,0.333
    conv1_1 = conv2d('conv1_1', X, weights['conv1_1'], biases['conv1_1'])
    conv1_2 = conv2d('conv1_2', conv1_1, weights['conv1_2'], biases['conv1_2'])
    pool1 = max_pool('pool1', conv1_2, k=2)
    norm1 = lrn('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, dropout)
    conv2_1 = conv2d('conv2_1', norm1, weights['conv2_1'], biases['conv2_1'])
    conv2_2 = conv2d('conv2_2', conv2_1, weights['conv2_2'], biases['conv2_2'])
    pool2 = max_pool('pool2', conv2_2, k=2)
    norm2 = lrn('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, dropout)
    conv3_1 = conv2d('conv3_1', norm2, weights['conv3_1'], biases['conv3_1'])
    conv3_2 = conv2d('conv3_2', conv3_1, weights['conv3_2'], biases['conv3_2'])
    conv3_3 = conv2d('conv3_3', conv3_2, weights['conv3_3'], biases['conv3_3'])
    pool3 = max_pool('pool3', conv3_3, k=2)
    norm3 = lrn('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, dropout)
    conv4_1 = conv2d('conv4_1', norm3, weights['conv4_1'], biases['conv4_1'])
    conv4_2 = conv2d('conv4_2', conv4_1, weights['conv4_2'], biases['conv4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, weights['conv4_3'], biases['conv4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)
    norm4 = lrn('norm4', pool4, lsize=4)
    norm4 = tf.nn.dropout(norm4, dropout)
    conv5_1 = conv2d('myconv5_1', norm4, weights['myconv5_1'], biases['myconv5_1'])
    conv5_2 = conv2d('myconv5_2', conv5_1, weights['myconv5_2'], biases['myconv5_2'])
    pool5 = max_pool('pool5', conv5_2,k=2)
    norm5 = lrn('norm5', pool5, lsize=4)
    norm5 = tf.nn.dropout(norm5, dropout)
    # 28x28x512 -> 28x28xnum_classes
    narrow5 = conv2d('narrow5',norm5,weights['narrow5'],biases['narrow5'])
    if use_bias:
        upsampled5 = tf.nn.conv2d_transpose(narrow5,weights['upsample5'],[batchsize,224,224,num_classes+1],[1,224/7,224/7,1],name='upsample5',padding='SAME') + tf.reshape(biases['upsample5'],[1,1,1,num_classes+1])
    else:
        upsampled5 = tf.nn.conv2d_transpose(narrow5,weights['upsample5'],[batchsize,224,224,num_classes+1],[1,224/7,224/7,1],name='upsample5',padding='SAME')
    tf.histogram_summary('outbias',biases['upsample5'])
    tf.histogram_summary('upsample_W',weights['upsample5'])
    net = scales['upsample5'] * tf.reshape(upsampled5,[batchsize,224 * 224,num_classes+1])
    outdict = {'upsample5' : upsampled5}
    if numfuse >= 1:
        narrow4 = conv2d('narrow4',norm4,weights['narrow4'],biases['narrow4'])
        if use_bias:
            upsampled4 = tf.nn.conv2d_transpose(narrow4,weights['upsample4'],[batchsize,224,224,num_classes+1],[1,224/14,224/14,1],name='upsample4',padding='SAME') + tf.reshape(biases['upsample4'],[1,1,1,num_classes+1])
        else:
            upsampled4 = tf.nn.conv2d_transpose(narrow4,weights['upsample4'],[batchsize,224,224,num_classes+1],[1,224/14,224/14,1],name='upsample4',padding='SAME')
        outdict['upsample4'] = upsampled4
        net += scales['upsample4'] * tf.reshape(upsampled4,[batchsize,224 * 224,num_classes+1])
    if numfuse == 2:
        narrow3 = conv2d('narrow3',norm3,weights['narrow3'],biases['narrow3'])
        if use_bias:
            upsampled3 = tf.nn.conv2d_transpose(narrow3,weights['upsample3'],[batchsize,224,224,num_classes+1],[1,224/28,224/28,1],name='upsample3',padding='SAME') + tf.reshape(biases['upsample3'],[1,1,1,num_classes+1])
        else:
            upsampled3 = tf.nn.conv2d_transpose(narrow3,weights['upsample3'],[batchsize,224,224,num_classes+1],[1,224/28,224/28,1],name='upsample3',padding='SAME')
        outdict['upsample3'] = upsampled3
        net += scales['upsample3'] * tf.reshape(upsampled3,[batchsize,224 * 224,num_classes+1])
    elif numfuse > 2:
        raise NotImplementedError
    outdict['net'] = net
    return outdict

def mkopt(X,Xgt,parameters,dropout,num_classes,batchsize,numfuse,bg_scale,alphas=None,di=None,reg_scale=0,use_bias=False):
    weights,biases = parameters
    outdict = arch(X,weights,biases,dropout,num_classes,batchsize,numfuse,alphas=alphas,di=di)
    gtexists = tf.less(tf.argmax(Xgt,2),num_classes) # 1 as the num_classes position indicates non-existing ground truth (0,num_classes-1) are the classes.
    fg_gt = tf.boolean_mask(Xgt,gtexists)
    bg_gt = tf.boolean_mask(Xgt,tf.logical_not(gtexists))
    fg_out = tf.boolean_mask(outdict['net'],gtexists)
    bg_out = tf.boolean_mask(outdict['net'],tf.logical_not(gtexists))
    loss_fg = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(fg_out,fg_gt))
    loss = loss_fg + (bg_scale * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(bg_out,bg_gt)))
    pos_is_accurate = tf.equal(tf.argmax(fg_out,1),tf.argmax(fg_gt,1))
    neg_is_accurate = tf.equal(tf.argmax(bg_out,1),tf.argmax(bg_gt,1))
    is_accurate = (pos_is_accurate,neg_is_accurate)
    loss_reg = tf.nn.l2_loss(weights['upsample5']) + tf.nn.l2_loss(weights['upsample4']) + tf.nn.l2_loss(weights['upsample3'])
    if use_bias:
        loss_reg += 500 * (tf.nn.l2_loss(biases['upsample5']) + tf.nn.l2_loss(biases['upsample4']) + tf.nn.l2_loss(biases['upsample3']))
    loss += reg_scale * loss_reg
    opt = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)
    vis_shape = [batchsize,224,224]
    return is_accurate,loss,opt,tf.reshape(tf.argmax(outdict['net'],2),vis_shape),tf.reshape(tf.argmax(Xgt,2),vis_shape),outdict,loss_fg,loss_reg

def savehdf(bgscale,nickname,splitid=None):
    batchsize = 40
    visualize = outer_vis(dataset,split,splitid=splitid)
    num_classes = len(params.possible_splits[splitid]['known'])
    modeldir,logdir = modeldir_of(nickname,trial,splitid,numfuse,dataset)
    sessname = str(splitid)
    with tf.Session() as sess:
        # Using a concrete batchsize because deconvolution had a problem with a ? dimension.
        _X = tf.placeholder(tf.float32,[batchsize,224,224,3])
        _pix = tf.placeholder(tf.float32,[batchsize,224 * 224,num_classes+1])
        _dropout = tf.placeholder(tf.float32)
        parameters = initialize(num_classes,numfuse)
        isaccurate,loss,optimizer,outimgs,inimgs,lossfg = mkopt(_X,_pix,parameters,_dropout,num_classes,batchsize,numfuse,bgscale,alphas,debug_info)
        pos_accurate,neg_accurate = isaccurate
        saver = tf.train.Saver(max_to_keep=50)
        if os.path.exists(params.root(modeldir)) and len(os.listdir(params.root(modeldir))) > 0:
            ckpt = tf.train.get_checkpoint_state(modeldir)
            t0 = int(ckpt.model_checkpoint_path.split("-")[-1])
            saver.restore(sess,ckpt.model_checkpoint_path)
        npyp = ({k :sess.run(v) for k,v in parameters[0].items()},{k :sess.run(v) for k,v in parameters[1].items()})
        dd.io.save(modeldir + "/best.hdf",npyp)
