'''
Alex Seewald 2016
aseewald@indiana.edu

Various helper functions which are either used by multiple files (e.g conv2d) 
or whose name is supposedly obvious enough that it doesn't need to take up space in the file.
'''
import numpy as np
import os
import re
import subprocess
import random
import pickle
import tensorflow as tf
from collections import OrderedDict
from skimage.io import imread,imsave
from skimage.color import gray2rgb
from scipy.misc import imresize
from skimage import img_as_float
import pandas as pd
import constants
if "sqlite" in constants.dbtypes:
    import sqlite3
if "postgres" in constants.dbtypes:
    import psycopg2
from params import params
import datasets

__author__ = "Alex Seewald"

# DATABASE RELATED
def dosql(stmt,only_log=False,log_if_fail=False,via_csv=False,whichdb="postgres",is_many=False):
    '''
    A wrapper around database inserts which will use both sqlite and postgres, for sake of duplication, if so configured.
    Rather than failing upon timeouts, this function also pushes SQL into loggs and moves on.
    Makes connection, does sql statement, closes connection.
    only_log means no database interaction will actually be done, just a todo list of sql inserts is appended to.
    log_if_fail means that a database insert fail will not cause the program to stop running, rather the todo list will get appends.
    '''
    if type(stmt) == dict:
        pg = stmt['pg']
        lite = stmt['lite']
    elif type(stmt) == list and is_many:
        # do an executemany.
        pass
    else:
        pg = lite = stmt
    if "sqlite" in constants.dbtypes or whichdb in ["sqlite","all"]:
        if only_log:
            if via_csv:
                idx = lite.find('(')
                lite = lite[idx+1:-1]
            with open('/data_b/aseewald/lite_wal.sql','a') as f:
                f.write(lite + ";\n")
        else: 
            try:
                conn = sqlite3.connect(params.db,timeout=300)
                cursor = conn.cursor()
                cursor.execute(lite)
                conn.commit()
                conn.close()
            except:
                if log_if_fail:
                    with open('/data_b/aseewald/lite_wal.sql','a') as f:
                        f.write(lite + ";\n")
                else:
                    raise IOError
    if "postgres" in constants.dbtypes or whichdb in ["postgres","all"]:
        if only_log:
            with open('pg_wal.sql','a') as f:
                f.write(pg + ";\n")
        else:
            try:
                conn = psycopg2.connect(**params.pg)
                cursor = conn.cursor()
                cursor.execute(pg)
                conn.commit()
                conn.close()
            except:
                if log_if_fail:
                    with open('/data_b/aseewald/lite_wal.sql','a') as f:
                        f.write(lite + ";\n")
                else:
                    raise IOError

def determine_checkpoint(modeldir,netfunc,feedfunc,numbatch):
    '''
    Do a binary search, starting at the end. If there aren't nans at a point in time, do some validation.

    Intended to be general across architectures, but just tested with the fully conv network so far.
    '''
    ckpt = tf.train.get_checkpoint_state(modeldir)
    ts = [x for x in os.listdir(modeldir) if x[-4:] != 'meta']
    random.shuffle(ts)
    for t in ts:
        print("Working on checkpoint={}".format(t))
        sess = tf.Session()
        net,accuracy = netfunc()
        saver = tf.train.Saver()
        saver.restore(sess,os.path.join(modeldir,t))
        for i in range(numbatch): 
            feed = feedfunc()
            netout,acc = sess.run([net,accuracy],feed)
            accs.append(acc)
        accuracies[t] = np.mean(accs)
        sess.close()
                
def normalize_unscaled_logits(tensor):
    '''
    tensor is a tensor of logits.
    The last dimension is supposed to be like "class".
    '''
    unscaled_probs = 1 / (1 + np.exp(-1 * tensor)) 
    scales = np.sum(unscaled_probs,axis=(len(tensor.shape)-1),keepdims=True)
    return unscaled_probs / scales

def readsql(query_stmt,whichdb="default",lowlevel=True,chunksize=None,conn=None):
    '''
    conn can be None in which case we build it up and tear it down here. if not None, it is persistent and passed in.
    '''
    closeit = False
    assert(whichdb in ["default","sqlite","postgres"])
    if "sqlite" in constants.dbtypes and (whichdb in ["sqlite","default"]):
        # make a readonly connection with some weird syntax.
        conn = sqlite3.connect("file:{}?mode=ro".format(params.read_db),uri=True,timeout=300)
        liteout = pd.read_sql(query_stmt,conn)
        conn.close()
        if (whichdb != "default") or len(constants.dbtypes) == 1: return liteout
    if "postgres" in constants.dbtypes and (whichdb in ["postgres","default"]):
        if conn is None: conn,closeit = psycopg2.connect(**params.pg),True
        if lowlevel:
            cursor = conn.cursor()
            cursor.execute(query_stmt)
            colnames = [desc[0] for desc in cursor.description]
            pgout = pd.DataFrame(cursor.fetchall(),columns=colnames)
        else:
            pgout = pd.read_sql(query_stmt,conn,chunksize=chunksize)
        if closeit: conn.close()
        if whichdb != "default" or len(constants.dbtypes) == 1: return pgout
    if whichdb == "default" and len(constants.dbtypes) > 1:
        if not liteout.equals(pgout) and not np.all(liteout.values == pgout.values): #two distinct conditions in case column names differ (interfaces with sqlite and psql working a little differently).
            print("Warning dbs do not match on query {}".format(query_stmt))
        return pgout
    else:
        return pgout

def xplatformtime(q):
    "generic interface for timestamps, in case using redundant dbs."
    return {'pg' : q('now()'), 'lite' : q("datetime('now')")}

def encode_imgname(imgname):
    '''
    For space-heavy pixgt, save gigabytes by having lower space usage.
    '''
    parts = os.path.split(imgname)
    if parts[0] == '':
        name = imgname.replace('COCO_','').replace('_000000','').replace('train2014','t').replace('val2014','v')
    else:
        name = os.path.join(parts[0],parts[1].replace('COCO_','').replace('_000000','').replace('train2014','t').replace('val2014','v'))
    return name.replace('.png','').replace('.jpg','') # remove extensions.

def decode_imgname(imgname):
    '''
    For space-heavy pixgt, save gigabytes by having lower space usage.
    '''
    parts = os.path.split(imgname)
    if parts[0] == '':
        return 'COCO_' + imgname.replace('t','train2014_000000').replace('v','val2014_000000')
    else:
        return os.path.join(parts[0],'COCO_' + parts[1].replace('t','train2014_000000').replace('v','val2014_000000'))

def insert_ifnotexists(query_stmt,insert_stmt,whichdb="postgres"):
    '''
    The name is self-explanatory.
    SQL should really have this built in.
    '''
    if len(readsql(query_stmt,whichdb=whichdb)) == 0:
        dosql(insert_stmt,whichdb=whichdb)
    else:
        print("Warning, did not execute this insert because matching data exists:\n{}\n{}".format(query_stmt,insert_stmt))

def addcat(split):
    for i,split in enumerate(params.possible_splits):
        for category in split['known']:
            dosql("INSERT INTO splitcats VALUES({},1,'{}',NULL)".format(i,category))
        for category in split['unknown']:
            dosql("INSERT INTO splitcats VALUES({},0,'{}',NULL)".format(i,category))

def possibly_convert_stats(df,imgstep):
    if len(df.keys()) == 2:
        df['tstep'] = np.repeat(np.arange(0,imgstep * len(df)//2,imgstep),2) #repeating because alteranting true and false selected rows.
        converted = True
    elif len(df.keys()) == 3:
        print("Already in the good format")
        converted = False
    else:
        assert(False)
    return df,converted

# TENSORFLOW RELATED.
def countnan(arr):
    return np.count_nonzero(np.isnan(arr))

def checknan(sess,parameters):
    weights,biases = parameters
    badcount = 0
    for k,v in weights.items():
        out = sess.run(v)
        if type(out) == tf.python.framework.ops.SparseTensorValue: #for the sparse parameters, extra information is returned. Discard it here.
            out = out.values
        if countnan(out) > 0:
            print("Warning nan weights for",k)
            badcount += 1
    for k,v in biases.items():
        if countnan(sess.run(v)) > 0:
            print("Warning nan biases for",k)
            badcount += 1
    if badcount == 0:
        print("No NAN values")
    return badcount == 0

# Wrapper functions for making my code not repeat the decisions, e.g. alpha value and stride types, that are *always* in effect.
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
def lrn(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
def leaky_relu(name,l_input,alpha=0.05):
    # if l_input[i,j..] is positive, then the non-leak case is true.
    # if l_input[i,j..] is negative, then the leak case is true.
    return tf.maximum(l_input,alpha * l_input)

def totensors(weights,trainable,extra_weights={},extra_biases={},xavier={ },hack_suffix=None):
    '''
    Trainable can be a dict of whether certain tensors are trainable, or it can simply be the value 'True' and then everything is considered trainable.
    If weights[0] or weights[1] is None, then a variable (weight or bias respectively) does not get created.
    '''
    # Trainable tuple values mean whether weights and biases are trainable, respectively.
    if trainable == True:
        trainable = {k : (True,True) for k in weights.keys()}
    elif type(trainable) == dict:
        for k,v in trainable.items():
            if (type(v) != tuple) and type(v) == bool:
                trainable[k] = (v,v) 
    if len(xavier) == 0:
        xavier = {k : False for k in weights.keys() }
    keys = set(xavier.keys())
    xavier_keys = set([k for k in xavier.keys() if xavier[k] == True])
    wkeys = set(weights.keys())
    assert(keys == wkeys)
    W,b = OrderedDict({}),OrderedDict({})
    for k in keys - xavier_keys:
        v = weights[k]
        if hack_suffix != None:
            if v[0] is not None: W[k] = tf.Variable(v[0],dtype=tf.float32,name="{}_weight_{}".format(k,hack_suffix),trainable=trainable[k][0])
            if v[1] is not None: b[k] = tf.Variable(v[1],dtype=tf.float32,name="{}_bias_{}".format(k,hack_suffix),trainable=trainable[k][1])
        else:
            if v[0] is not None: W[k] = tf.Variable(v[0],dtype=tf.float32,name="{}_weight".format(k),trainable=trainable[k][0])
            if v[1] is not None: b[k] = tf.Variable(v[1],dtype=tf.float32,name="{}_bias".format(k),trainable=trainable[k][1])
    # Don't bother with xavier initialization for weights and biases that don't have counterpart.
    for k,v in extra_weights.items():
        if hack_suffix != None:
            if v[0] is not None: W[k] = tf.Variable(v,dtype=tf.float32,name="{}_weight_{}".format(k,hack_suffix))
        else:
            if v[0] is not None: W[k] = tf.Variable(v,dtype=tf.float32,name="{}_weight".format(k))
    for k,v in extra_biases.items():
        if hack_suffix != None:
            if v[1] is not None: b[k] = tf.Variable(v,dtype=tf.float32,name="{}_bias_{}".format(k,hack_suffi))
        else:
            if v[1] is not None: b[k] = tf.Variable(v,dtype=tf.float32,name="{}_bias".format(k))
    print("USING XAVIER INITIALZIATION on ", [k for k in xavier.keys() if xavier[k] == True])
    for k in xavier_keys:
        v = weights[k]
        xavinit = tf.contrib.layers.xavier_initializer() if len(v[0].shape) == 2 else tf.contrib.layers.xavier_initializer_conv2d()
        if v[0] is not None: W[k] = tf.get_variable(k, shape=v[0].shape,initializer=xavinit,dtype=tf.float32)
        if v[1] is not None: b[k] = tf.Variable(v[1],dtype=tf.float32)
    return(W,b)

def determine_uninitialized(sess):
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars

def extract(modeldir,outname,splitid=None):
    '''
    Restores from a tensorflow checkpoint and saves paramters as an npy file for later use.
    Splitid is optional because embed initialization doesn't really have a notion of classes,
    whereas to avoid cheating during training vanilla initialization should share a splitid.

    Examples:
        extract('models/arch_<x>/','vanilla',splitid=2)
        extract('loctrain/arch_<x>/','embed')
    '''
    with tf.Session() as sess:
        weights,biases = initialize(4,"vggnet",only_fromnpy=True)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(modeldir)
        saver.restore(sess,ckpt.model_checkpoint_path)
        npyweights = {k: sess.run(v) for k,v in weights.items()}
        npybiases = {k: sess.run(v) for k,v in biases.items()}
        todump = {k : (npyweights[k],npybiases[k]) for k in npyweights.keys()}
        if splitid:
            np.save('{}_{}.npy'.format(outname,splitid),todump)
        else:
            np.save('{}.npy'.format(outname),todump)

def arr_isintegral(arr):
    return np.equal(np.mod(arr, 1), 0)

def onehot(vals,ndistinct):
    return np.equal.outer(vals,np.arange(ndistinct))

def vggrand(scale):
    '''
    A fully random vggnet.
    '''
    weights = {}
    weights['conv1_1'] = [scale * np.random.randn(3,3,3,64),scale * np.random.randn(64)]
    weights['conv1_2'] = [scale * np.random.randn(3,3,64,64),scale * np.random.randn(64)]
    weights['conv2_1'] = [scale * np.random.randn(3,3,64,128),scale * np.random.randn(128)]
    weights['conv2_2'] = [scale * np.random.randn(3,3,128,128),scale * np.random.randn(128)]
    weights['conv3_1'] = [scale * np.random.randn(3,3,128,256),scale * np.random.randn(256)]
    weights['conv3_2'] = [scale * np.random.randn(3,3,256,256),scale * np.random.randn(256)]
    weights['conv3_3'] = [scale * np.random.randn(3,3,256,256),scale * np.random.randn(256)]
    weights['conv4_1'] = [scale * np.random.randn(3,3,256,512),scale * np.random.randn(512)]
    weights['conv4_2'] = [scale * np.random.randn(3,3,512,512),scale * np.random.randn(512)]
    weights['conv4_3'] = [scale * np.random.randn(3,3,512,512),scale * np.random.randn(512)]
    weights['conv5_1'] = [scale * np.random.randn(3,3,512,512),scale * np.random.randn(512)]
    weights['conv5_2'] = [scale * np.random.randn(3,3,512,512),scale * np.random.randn(512)]
    weights['conv5_3'] = [scale * np.random.randn(3,3,512,512),scale * np.random.randn(512)]
    weights['fc6'] = [scale * np.random.randn(25088, 4096), scale * np.random.randn(4096)]
    weights['fc7'] = [scale * np.random.randn(4096, 4096), scale * np.random.randn(4096)]
    weights['fc8'] = [scale * np.random.randn(4096, 1000), scale * np.random.randn(1000)]
    return weights

# IMAGE RELATED
def imread_wrap(path,tmp=False):
    '''imread that enforces a (224,224,3) RGB output.
       if tmp is true, don't do any caching.
    '''
    if tmp:
        img = imread(path)
        if len(img.shape) == 2:
            img = gray2rgb(img)
        img = img_as_float(imresize(img,(224,224)))
    d,f = os.path.split(path)
    if d == params.root("val_images"):
        quickpath = params.root("val_squareimgs")
    elif d == params.root("val_patches"):
        quickpath = params.root("val_squarepatches")
    elif d == params.root("train_images"):
        quickpath = params.root("train_squareimgs")
    elif d == params.root("train_patches"):
        quickpath = params.root("train_squarepatches")
    elif d == params.root("val_candidateimgs"):
        quickpath = params.root("val_squarecandidateimgs")
    elif d == params.root("debug"):
        quickpath = params.root("train_quickdebug")
    else:
        quickpath = d + '-cache'
    qname = os.path.join(quickpath,f)
    if not os.path.exists(quickpath):
        subprocess.call(["mkdir",quickpath])
    if not os.path.exists(qname):
        img = imread(path)
        if len(img.shape) == 2:
            img = gray2rgb(img)
        img = img_as_float(imresize(img,(224,224)))
        imsave(qname,img)
        return img
    else:
        return img_as_float(imread(qname))

def cocofmt(istrain,imgnum):
    digits = len(imgnum) #imgnum is a string.
    coconumdigits = 12
    t = "train" if istrain else "val"
    return "COCO_{}2014_".format(t) + "0" * (coconumdigits - digits) + imgnum + ".jpg"

def fullOf(x,istrain,imgid_fmt,dataset):
    if dataset == 'COCO':
        s = "train" if istrain else "val"
        d = params.root("{}_images".format(s))
        if imgid_fmt:
            imgnum = x.split("_")[0]
            digits = len(str(imgnum))
            return os.path.join(d,cocofmt(istrain,imgnum))
        else:
            fname = s.split("_")[0:2] + ".jpg"
            return os.path.join(d,fname)
    elif dataset == 'pascal':
        return os.path.join(params.root("val_images/"+ "_".join(x.split("_")[0:2]) + ".jpg"))

def imgname_cache(splitid,variety,dataset):
    '''
    This stores the locations of image data that the sample functions should look at.
    A time for space tradeoff that is well worth it.
    '''
    def ctxpath_decode(dname,inclusion="split",dataset='COCO'):
        '''
        Inclusion is an argument whose value inclusion="split" means keep only the known classes and value inclusion="notsplit" means keep the unknown classes, and "all"
        means keep all classes.

        dname is the directory name containing the imgnames to be found, depending on whether using expanded bounding boxes.
        '''
        if dataset == 'COCO':
            idir = "train" if dname.split("_")[0] == "train" else "val"
            isexpanded = dname.split("_")[1] in ["ctxpatches","xlctxpatches"]
            isxl = dname.split("_")[1] == "xlctxpatches"
            #bboxs = readsql("SELECT * FROM perfect_bbox INNER JOIN imgsize ON perfect_bbox.imgname = '{4}' || imgsize.imgname || '.jpg' WHERE perfect_bbox.patchname LIKE '{0}%' AND ((maxy - miny) / height) > {1} AND ((maxx - minx) / width) > {1} AND isxl = {2} AND isexpanded = {3}".format(params.root(dname),1/3,int(isxl),int(isexpanded),params.root(idir + "_images/")))
            bboxs = readsql("SELECT * FROM perfect_bbox INNER JOIN imgsize ON perfect_bbox.imgname = '{4}' || imgsize.imgname || '.jpg' WHERE perfect_bbox.patchname LIKE '{0}%' AND ((maxy - miny) / height::float) > {1} AND ((maxx - minx) / width::float) > {1} AND isxl = {2} AND isexpanded = {3}".format(params.root(dname),1/3,int(isxl),int(isexpanded),params.root(idir + "_images/")))
            names = bboxs['patchname'].values
            labels = np.array(["_".join(os.path.split(name)[1].split("_")[1:-1]) for name in names])
            if inclusion == "split":
                which = [(label in params.possible_splits[splitid]['known']) for label in labels]
            elif inclusion == "notsplit":
                which = [(label not in params.possible_splits[splitid]['known']) for label in labels]
            elif inclusion == "all":
                which = [True for label in labels]
            which = np.array(which)
            # further filter by minimum allowed size and whether bbox exists.
            return(names[which],labels[which])
        elif dataset == 'pascal':
            cans = readsql("SELECT * FROM candidate_bbox NATURAL JOIN ground_truth WHERE dataset = 'pascal'") #not yet tested but seems okay.
            knowns = readsql("select * from splitcats where dataset = 'COCO' AND splitid = {} AND seen = 1".format(splitid))
            for equiv in datasets.equivalences['coco,pascal']:
                knowns['category'].replace(equiv['coco'],equiv['pascal'],inplace=True)
            mask = cans['classname'].isin(knowns['category'])
            if inclusion == "notsplit":
                mask = np.logical_not(mask)
                print("np.mean(mask)={}".format(np.mean(mask)))
            elif inclusion == "split":
                print("np.mean(mask)={}".format(np.mean(mask)))
            else:
                assert(False)
            mask = np.logical_and(cans['classname'] != 'None',mask)
            cans = cans[mask]
            cans = cans.reset_index(drop=True)
            return (cans['imgname'],cans['canid']),cans['classname']
    if not os.path.exists('cache'):
        subprocess.call(["mkdir","cache"])
    if variety == "train_normal": #this is what I'm using now, got a better idea than simply bigger bounding boxes for context.
        fname = 'cache/train_normal_{}.pkl'.format(splitid) if dataset == 'COCO' else 'cache/pascal-train_normal_{}.pkl'.format(splitid)
        if not os.path.exists(fname):
            train_names = ctxpath_decode("train_patches","split",dataset)
            pickle.dump(train_names,open(fname,'wb'))
        else:
            train_names = pickle.load(open(fname,'rb'))
        return train_names
    if variety == "unseen_normal":
        fname = 'cache/unseen_normal_{}.pkl'.format(splitid) if dataset == 'COCO' else 'cache/pascal-unseen_normal_{}.pkl'.format(splitid)
        if not os.path.exists(fname):
            unseen_names = ctxpath_decode("train_patches","notsplit",dataset)
            pickle.dump(unseen_names,open(fname,'wb'))
        else:
            unseen_names = pickle.load(open(fname,'rb'))
        return unseen_names
    elif variety == "test":# Testing on candidates
        if not os.path.exists('cache/test_{}.pkl'.format(splitid)):
            qstr = ",".join(["'{}'".format(cat) for cat in params.possible_splits[splitid]['unknown']])
            print(qstr)
            imgnames,canids,labels = readsql("SELECT imgname,canid,classname FROM ground_truth WHERE classname IN ({})".format(qstr)).values.T
            print(len(imgnames))
            # making the name ready for imread in the val_candidateimgs dir.
            imgnames = ["_".join([imgnames[i],str(0),params.candidate_method,str(canids[i])]) + ".jpg" for i in range(len(imgnames))]
            test_names = (imgnames,labels)
            pickle.dump(test_names,open('cache/test_{}.pkl'.format(splitid),'wb'))
        else:
            test_names = pickle.load(open('cache/test_{}.pkl'.format(splitid),'rb'))
        return test_names
    elif variety == "curriculum":
        pass #
    elif variety == "testperfect_unseen":
        if not os.path.exists('cache/unseenPerfect_{}.pkl'.format(splitid)):
            unseen_names = ctxpath_decode("val_ctxpatches","notsplit",dataset)
            pickle.dump(unseen_names,open("cache/unseenPerfect_{}.pkl".format(splitid),'wb'))
        else:
            unseen_names = pickle.load(open("cache/unseenPerfect_{}.pkl".format(splitid),'rb'))
        return unseen_names
    elif variety == "testperfect_seen":
        if not os.path.exists('cache/seenPerfect_{}.pkl'.format(splitid)):
            seen_names = ctxpath_decode("val_ctxpatches","split") 
            pickle.dump(seen_names,open("cache/seenPerfect_{}.pkl".format(splitid),'wb'))
        else:
            seen_names = pickle.load(open("cache/seenPerfect_{}.pkl".format(splitid),'rb'))
        return seen_names
    else:
        assert(False), "Unknown variety {}".format(variety)
 
def auc(precision,recall):
    '''
    Takes the sorted precision/recall arrays and calculates area with trapevoidal rule.
    '''
    return np.dot(precision[:-1],np.diff(recall))

def sample_img(N,splitid,variety="train_normal",imgname=None,include_saliency=False,val_candidates=None,full_img=True,dataset='COCO'):
    '''
    Using nested functions here to avoid repeatedly doing all the work to get names of possible things to sample.
    The variable 'static' is bound by the outer function call, emulating C's idea of static variables.
    '''
    all_names,all_labels = imgname_cache(splitid,variety,dataset)
    def shapeOf(bbox):
        return readsql("SELECT height,width FROM imgsize WHERE imgname = '{}'".format(os.path.splitext(os.path.split(bbox['imgname'].ix[0])[1])[0]))
    def boundcond(num):
        return (0 <= num <= 224)
    def adjust(bbox):
        shape = shapeOf(bbox)
        if len(shape) == 0:
            return False
        bbox['miny'],bbox['maxy'] = bbox['miny'].ix[0] * (224 / shape['height'].ix[0]),bbox['maxy'].ix[0] * (224 / shape['height'].ix[0])
        bbox['minx'],bbox['maxx'] = bbox['minx'].ix[0] * (224 / shape['width'].ix[0]),bbox['maxx'].ix[0] * (224 / shape['width'].ix[0])
        if (bbox['maxy'].ix[0] > 224) and np.allclose(bbox['maxy'].ix[0],224,rtol=1e-2):
            bbox['maxy'].ix[0] = 224
        if (bbox['maxx'].ix[0] > 224) and np.allclose(bbox['maxx'].ix[0],224,rtol=1e-2):
            bbox['maxx'].ix[0] = 224
        try:
            assert(boundcond(bbox['miny'].ix[0]))
            assert(boundcond(bbox['maxy'].ix[0]))
            assert(boundcond(bbox['minx'].ix[0]))
            assert(boundcond(bbox['maxx'].ix[0]))
        except:
            print("assert fail")
            input()
        return bbox
    # I think I need to add 
    if variety in ["train_normal","unseen_normal","testperfect_normal"]:
        if dataset == 'COCO':
            dirname = params.root('train_patches')
        elif dataset == 'pascal':
            dirname = params.root('val_candidateimgs')
        def bboxOf(name,adjusted=True):
            if dataset == 'COCO':
                bb = readsql("SELECT miny,maxy,minx,maxx,imgname FROM perfect_bbox WHERE patchname = '{}' AND isexpanded = 0".format(os.path.join(dirname,name)))
            else:
                bb = readsql("SELECT min AS miny,maxy,minx,maxx,imgname from candidate_bbox WHERE dataset = 'pascal' AND imgname = '{}' AND canid = {}".format(name[0],name[1]))
            if adjusted:
                bb = adjust(bb)
            return bb
    elif variety == "test":
        def bboxOf(name,adjusted=True):
            canid = os.path.splitext(name.split("_")[5])[0]
            name = name.split("_")[0:3] + ".jpg"
            bb = readsql("SELECT miny,maxy,minx,maxx FROM candidate_bbox WHERE imgname = '{}' AND canid = {}".format(name,canid))
            if adjusted:
                bb = adjust(bb)
            return bb
        if dataset == 'COCO':
            dirname = params.root("val_candidateimgs")
        elif dataset == 'pascal':
            dirname = params.root('val_candidateimgs')
    if variety in ["train_normal","unseen_normal","train_ctx","unseen_ctx","train_xlctx","unseen_xlctx"]:
        full_flag,imgid_fmt = True,True
    elif variety in ["testperfect_seen","testperfect_unseen"]:
        full_flag,imgid_fmt = False,True
    elif variety == "test":
        imgid_fmt = False,False
    else:
        print("Unknown variety {}".format(variety))
        sys.exit(1)
    def inner(tomatch=None,specified=None):
        '''
        Train happens with train_ctxpatches.
        Discovery happens with candidates. include_saliency can happen with candidates.
        tomatch is a list of categories.
        '''
        if dataset == 'pascal':
            assert(not include_saliency)
            names,canids,labels,bboxs = [],[],[],[]
            if not tomatch is None:
                while len(names) < N:
                    i = len(names)
                    if specified is None:
                        if random.random() < constants.negprop:
                            idx = random.choice(range(len(all_names[0])))
                        else:
                            idx = random.choice(np.where(all_labels == tomatch[i])[0])
                    else:
                        idx = np.argmax(all_names == specified[i])
                    name,canid,label = all_names[0][idx],all_names[1][idx],all_labels[idx]
                    bbox = bboxOf((name,canid))
                    if bbox is False:
                        continue
                    names.append(os.path.join(dirname,name + "_objectness_" + str(canid) + ".jpg"))
                    labels.append(label)
                    canids.append(canid)
                    bboxs.append(bbox[['miny','maxy','minx','maxx']].values)
            else:
                while len(names) < N:
                    idx = random.choice(range(len(all_names[0])))
                    name,canid = all_names[0][idx],all_names[1][idx]
                    bbox = bboxOf((name,canid))
                    if bbox is False:
                        continue
                    label = all_labels[idx]
                    canids.append(canid)
                    names.append(os.path.join(dirname,name + "_objectness_" + str(canid) + ".jpg"))
                    labels.append(label)
                    bboxs.append(bbox[['miny','maxy','minx','maxx']].values)
            names,labels = np.array(names),np.array(labels)
            imgs = np.array([imread_wrap(os.path.join(dirname,x)) for x in names])
            if full_img:
                full_imgs = np.array([imread_wrap(fullOf(os.path.split(x)[1],full_flag,imgid_fmt,dataset)) for x in names])
                assert(np.max(imgs) <= 1 and np.min(imgs) >= 0)
                assert(np.max(full_imgs) <= 1 and np.min(full_imgs) >= 0)
                return(imgs,full_imgs,np.array(bboxs).squeeze(),labels,names)
            else:
                return(imgs,labels,names)
        if dataset == 'COCO':
            if include_saliency:
                sal_imgs,imgs,labels = [],[],[]
                conn = sqlite3.connect(params.read_db,timeout=300)
                while (len(imgs) < N) and (len(sal_imgs) < N):
                    candidates = val_candidates.sample(N)
                    try: # some of these may fail.
                        for rowid,row in candidates.iterrows():
                            imgs.append(imread_wrap(os.path.join(params.root('val_images'),row['imgname'] + ".jpg")))
                            imgcans = val_candidates[val_candidates['imgname'] == row['imgname']]
                            canid = np.argmax(np.all((row == imgcans).values,axis=1))
                            cat = pd.read_sql("SELECT classname FROM ground_truth WHERE imgname = '{}' AND canid = {}".format(row['imgname'],canid),conn)['classname'].values[0]
                            labels.append(cat)
                            sal_imgs.append(imresize(imread(params.root("val_saliency/") + row['imgname'] + ".jpg")[row['miny']:row['maxy'],row['minx']:row['maxx']],(224,224)))
                    except:
                        imgs,labels,sal_imgs = [], [], []
                        continue
                    conn.close()
                return(np.array(imgs),np.array(labels),np.array(sal_imgs))
            else:
                names,labels,bboxs = [],[],[]
                if not tomatch is None:
                    while len(names) < N:
                        i = len(names)
                        if specified is None:
                            if random.random() < constants.negprop:
                                idx = random.choice(range(len(all_names)))
                            else:
                                idx = random.choice(np.where(all_labels == tomatch[i])[0])
                        else:
                            idx = np.argmax(all_names == specified[i])
                        name,label = all_names[idx],all_labels[idx]
                        bbox = bboxOf(name)
                        if bbox is False:
                            continue
                        names.append(os.path.join(dirname,name))
                        labels.append(label)
                        bboxs.append(bbox[['miny','maxy','minx','maxx']].values)
                else:
                    while len(names) < N:
                        idx = random.choice(range(len(all_names)))
                        name = all_names[idx]
                        bbox = bboxOf(name)
                        if bbox is False:
                            continue
                        label = all_labels[idx]
                        names.append(os.path.join(dirname,name))
                        labels.append(label)
                        bboxs.append(bbox[['miny','maxy','minx','maxx']].values)
                names,labels = np.array(names),np.array(labels)
                imgs = np.array([imread_wrap(os.path.join(dirname,x)) for x in names])
                if full_img:
                    full_imgs = np.array([imread_wrap(fullOf(os.path.split(x)[1],full_flag,imgid_fmt,dataset)) for x in names])
                    assert(np.max(imgs) <= 1 and np.min(imgs) >= 0)
                    assert(np.max(full_imgs) <= 1 and np.min(full_imgs) >= 0)
                    return(imgs,full_imgs,np.array(bboxs).squeeze(),labels,names)
                else:
                    return(imgs,labels,names)
    return inner

def center_transform(imgs,names=None,bboxs=[],smallimgs=None,verify=False,zoom_t="out"):
    '''
    this interface could be improved a bit. bboxs must be provided but is after "names" which is optional, to not break previous code.
    '''
    def full_centered(Xfull,bbox):
        '''
        Needs to make a version of Xfull where X is in center and everything else is scaled accordingly.
        '''
        H,W = Xfull.shape[0],Xfull.shape[1]
        ymean,xmean = (bbox[0] + bbox[1]) / 2,(bbox[2] + bbox[3]) / 2
        hshift,wshift = int((H / 2) - ymean), int((W / 2) - xmean)
        alt_bbox = np.zeros(4)
        if zoom_t == "out":
            prop_hshift,prop_wshift = ((abs(hshift)+224)/224),((abs(wshift)+224)/224)
            wblack = np.zeros((H,2 * abs(wshift),3))
            alt_bbox[0:2] = (bbox[0:2] + hshift) / prop_hshift
            alt_bbox[2:4] = (bbox[2:4] + wshift) / prop_wshift
            if wshift > 0: #if object candidate is left of center, pad left with zeros.
                Xblack_x = np.hstack((wblack,Xfull))
            elif wshift < 0: #if object candidate is left of center, pad left with zeros.
                Xblack_x = np.hstack((Xfull,wblack))
            else: Xblack_x = Xfull
            hblack = np.zeros((2 * abs(hshift),Xblack_x.shape[1],3))
            if hshift > 0: Xout = np.vstack((hblack,Xblack_x))
            elif hshift < 0: Xout = np.vstack((Xblack_x,hblack))
            else: Xout = Xblack_x
            # imresize brings it back to integers, so i have to say go back to [0,1] floats.
        elif zoom_t == "in":
            pass
        return img_as_float(imresize(Xout,Xfull.shape)),alt_bbox
    outs,alt_bboxs = [],[]
    if len(imgs.shape) == 3: #in case just a single image
        return full_centered(imgs,np.array(bboxs))
    for i in range(len(imgs)):
        fc = full_centered(imgs[i],bboxs[i])
        outs.append(fc[0])
        alt_bboxs.append(fc[1])
        if verify: # verifying things are okay by looking at plots.
            fig,axes = plt.subplots(2)
            axes[0].imshow(outs[i])
            axes[0].axhline(112)
            axes[0].axvline(112)
            axes[1].imshow(smallimgs[i])
            axes[1].axhline(112)
            axes[1].axvline(112)
            plt.show() 
            plt.close()
    return np.array(outs),alt_bboxs

def transfer_exclude(df,train_dataset,splitid,dataset):
    '''
    When using arch trained on one dataset and applied to another.
    '''
    excluded_cats = readsql("select category from exclusions where source_dataset = '{}' AND splitid = {} AND target_dataset = '{}'".format(train_dataset,splitid,dataset))
    if 'classname' in df.keys():
        dfk = 'classname'
    else:
        dfk = 'category'
    mask = np.logical_not(df[dfk].isin(excluded_cats['category']))
    return df[mask]

# MISCELLANOUS
def chunks(l, n):
    '''Yield successive n-sized chunks from l'''
    for i in range(0, len(l), n):
        yield l[i:i+n]

def floatserial(L,sigfigs):
    if type(L) == list:
        L = np.array(L)
    if len(L.shape) == 1:
        return ','.join(['%.{}f'.format(sigfigs) % num for num in L])
    elif len(L.shape) == 2:
        return '['+ ','.join(['[' + ','.join(['%.{}f'.format(sigfigs) % num for num in row]) + ']' for row in L]) + ']'

def floateval(stringin):
    "Inverse of floatserial"
    return np.array(eval('[' + stringin + ']'))

def affinity_outfmt(x,splitid,nickname,num_candidates,even,perfect):
    return params.root("kernels/{}_{}_{}_{}_{}_{}.pkl".format(x,splitid,nickname,num_candidates,even,perfect))
