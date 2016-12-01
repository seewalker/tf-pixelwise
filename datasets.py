'''
Alex Seewald 2016
aseewald@indiana.edu

Contains code that is specific to the datasets.
'''
import os
import pickle

__author__ = "Alex Seewald"

# This is manually determined to be just like in [1].
voc2008 = [{
    'known' : ['bicycle','bird','boat','bottle','bus','cat','chair','dingingtable',
               'dog','horse','person','pottedplant','sheep','sofa','train'],
    'unknown' : ['aeroplane','car','cow','motorbike','tvmonitor'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison" 
}, {
    'known' : ['aeroplane', 'boat', 'bottle','bus','car','cat','cow','diningtable','dog','horse',
               'motorbike','person','pottedplant','sheep','tvmonitor'],
    'unknown' : ['bicycle','bird','chair','sofa','train'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison" 
}, {
    'known' : ['bicycle','bird','car','cat','cow','dog','horse','person','pottedplant','sheep'],
    'unknown' : ['aeroplane','boat','bottle','bus','chair','diningtable','motorbike','sofa','train','tvmonior'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison" 
}, {
    'known' : ['bird','boat','car','cat','chair','cow','diningtable','horse','pottedplant','tvmonitor'],
    'unknown' : ['aeroplane','bicycle','bottle','bus','dog','motorbike','person',
                 'sheep', 'sofa','train'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison" 
}, {
    'known' : ['aeroplane','bicycle','bird','person','sheep'],
    'unknown' : ['boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                 'horse','motorbike','pottedplant','train','tvmonitor'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison" 
}, {
    'known' : ['car','chair','diningtable','sheep','sofa'],
    'unknown' : ['aeroplane','bicycle','bird','boat','bottle','bus','cat','cow','dog',
                 'horse','motorbike','person','pottedplant','train','tvmonitor'],
    'significance' : "Same class from Object Graph paper, for sake of exact comparison"}]

coco_allcats = [u'baseball glove', u'orange', u'surfboard', u'handbag', u'backpack', u'bird', u'cell phone', u'tie', u'airplane', u'apple', u'banana', u'baseball bat', u'bear', u'bed', u'bench', u'bicycle', u'boat', u'book', u'bottle', u'bowl', u'broccoli', u'bus', u'cake', u'car', u'carrot', u'cat', u'chair', u'clock', u'couch', u'cow', u'cup', u'dining table', u'dog', u'donut', u'elephant', u'fire hydrant', u'fork', u'frisbee', u'giraffe', u'hair drier', u'horse', u'hot dog', u'keyboard', u'kite', u'knife', u'laptop', u'microwave', u'motorcycle', u'mouse', u'oven', u'parking meter', u'person', u'pizza', u'potted plant', u'refrigerator', u'remote', u'sandwich', u'scissors', u'sheep', u'sink', u'skateboard', u'skis', u'snowboard', u'spoon', u'sports ball', u'stop sign', u'suitcase', u'teddy bear', u'tennis racket', u'toaster', u'toilet', u'toothbrush', u'traffic light', u'train', u'truck', u'tv', u'umbrella', u'vase', u'wine glass', u'zebra']

equivalences = { 
    'coco,pascal' : [
        {'coco' : 'airplane', 'pascal' : 'aeroplane'},
        {'coco' : 'dining table', 'pascal' : 'diningtable'},
        {'coco' : 'potted plant', 'pascal' : 'pottedplant'},
        {'coco' : 'tv', 'pascal' : 'tvmonitor'},
        {'coco' : 'motorcycle', 'pascal' : 'motorbike'}
    ]
}
cname = 'clust_splits.pkl'
if not os.path.exists(cname):
    print("Run py2.scene_cluster to get splits based on co-occurence clustering")
    coco_clust = None
else:
    coco_clust = pickle.load(open(cname,'rb'))

# This is copy and pasted from an ipython session of randomly determining cateogories.
coco = [{'known' : [u'baseball glove', u'orange', u'surfboard', u'handbag', u'backpack', u'bird', u'cell phone', u'tie'],
    'unknown' : [u'airplane', u'apple', u'banana', u'baseball bat', u'bear', u'bed', u'bench', u'bicycle', u'boat', u'book', u'bottle', u'bowl', u'broccoli', u'bus', u'cake', u'car', u'carrot', u'cat', u'chair', u'clock', u'couch', u'cow', u'cup', u'dining table', u'dog', u'donut', u'elephant', u'fire hydrant', u'fork', u'frisbee', u'giraffe', u'hair drier', u'horse', u'hot dog', u'keyboard', u'kite', u'knife', u'laptop', u'microwave', u'motorcycle', u'mouse', u'oven', u'parking meter', u'person', u'pizza', u'potted plant', u'refrigerator', u'remote', u'sandwich', u'scissors', u'sheep', u'sink', u'skateboard', u'skis', u'snowboard', u'spoon', u'sports ball', u'stop sign', u'suitcase', u'teddy bear', u'tennis racket', u'toaster', u'toilet', u'toothbrush', u'traffic light', u'train', u'truck', u'tv', u'umbrella', u'vase', u'wine glass', u'zebra'],
    'significance' : "Split 0: few randomly generated knowns."},
    {'known' : [u'microwave', u'tennis racket', u'truck', u'toaster', u'train', u'scissors', u'vase', u'cell phone'],
     'unknown' : [u'airplane', u'apple', u'backpack', u'banana', u'baseball bat', u'baseball glove', u'bear', u'bed', u'bench', u'bicycle', u'bird', u'boat', u'book', u'bottle', u'bowl', u'broccoli', u'bus', u'cake', u'car', u'carrot', u'cat', u'chair', u'clock', u'couch', u'cow', u'cup', u'dining table', u'dog', u'donut', u'elephant', u'fire hydrant', u'fork', u'frisbee', u'giraffe', u'hair drier', u'handbag', u'horse', u'hot dog', u'keyboard', u'kite', u'knife', u'laptop', u'motorcycle', u'mouse', u'orange', u'oven', u'parking meter', u'person', u'pizza', u'potted plant', u'refrigerator', u'remote', u'sandwich', u'sheep', u'sink', u'skateboard', u'skis', u'snowboard', u'spoon', u'sports ball', u'stop sign', u'suitcase', u'surfboard', u'teddy bear', u'tie', u'toilet', u'toothbrush', u'traffic light', u'tv', u'umbrella', u'wine glass', u'zebra'],
    'significance' : "Split 1: few randomly generated knowns." },
    {'known' : [u'wine glass', u'baseball bat', u'carrot', u'truck', u'umbrella', u'banana', u'horse', u'cat', u'potted plant', u'dining table', u'sink', u'toothbrush', u'bear', u'surfboard', u'airplane', u'bird', u'sports ball', u'book', u'motorcycle', u'cake'],
     'unknown' : [u'apple', u'backpack', u'baseball glove', u'bed', u'bench', u'bicycle', u'boat', u'bottle', u'bowl', u'broccoli', u'bus', u'car', u'cell phone', u'chair', u'clock', u'couch', u'cow', u'cup', u'dog', u'donut', u'elephant', u'fire hydrant', u'fork', u'frisbee', u'giraffe', u'hair drier', u'handbag', u'hot dog', u'keyboard', u'kite', u'knife', u'laptop', u'microwave', u'mouse', u'orange', u'oven', u'parking meter', u'person', u'pizza', u'refrigerator', u'remote', u'sandwich', u'scissors', u'sheep', u'skateboard', u'skis', u'snowboard', u'spoon', u'stop sign', u'suitcase', u'teddy bear', u'tennis racket', u'tie', u'toaster', u'toilet', u'traffic light', u'train', u'tv', u'vase', u'zebra'],
    'significance' : "Split 2: more randomly generated unknowns than knowns." },
    {'known' : [u'surfboard', u'bicycle', u'frisbee', u'baseball bat', u'fire hydrant', u'scissors', u'snowboard', u'umbrella', u'dining table', u'train', u'backpack', u'bowl', u'tennis racket', u'sheep', u'traffic light', u'pizza', u'vase', u'potted plant', u'tv', u'sink'],
     'unknown' : [u'airplane', u'apple', u'banana', u'baseball glove', u'bear', u'bed', u'bench', u'bird', u'boat', u'book', u'bottle', u'broccoli', u'bus', u'cake', u'car', u'carrot', u'cat', u'cell phone', u'chair', u'clock', u'couch', u'cow', u'cup', u'dog', u'donut', u'elephant', u'fork', u'giraffe', u'hair drier', u'handbag', u'horse', u'hot dog', u'keyboard', u'kite', u'knife', u'laptop', u'microwave', u'motorcycle', u'mouse', u'orange', u'oven', u'parking meter', u'person', u'refrigerator', u'remote', u'sandwich', u'skateboard', u'skis', u'spoon', u'sports ball', u'stop sign', u'suitcase', u'teddy bear', u'tie', u'toaster', u'toilet', u'toothbrush', u'truck', u'wine glass', u'zebra'],
    'significance' : "Split 3: more randomly generated unknowns than knowns."},
    {'known' : [u'backpack', u'banana', u'baseball bat', u'baseball glove', u'bear', u'bed', u'bench', u'bird', u'boat', u'bowl', u'broccoli', u'cat', u'clock', u'couch', u'cup', u'dog', u'donut', u'fire hydrant', u'frisbee', u'hot dog', u'knife', u'laptop', u'microwave', u'mouse', u'orange', u'oven', u'person', u'pizza', u'refrigerator', u'sheep', u'sink', u'snowboard', u'spoon', u'sports ball', u'stop sign', u'suitcase', u'toilet', u'traffic light', u'tv', u'vase'],
     'unknown' : [u'train', u'airplane', u'carrot', u'elephant', u'handbag', u'car', u'wine glass', u'bicycle', u'bottle', u'toaster', u'skateboard', u'fork', u'zebra', u'tennis racket', u'cake', u'truck', u'book', u'sandwich', u'keyboard', u'motorcycle', u'giraffe', u'horse', u'scissors', u'toothbrush', u'parking meter', u'hair drier', u'remote', u'bus', u'kite', u'apple', u'umbrella', u'skis', u'dining table', u'surfboard', u'cow', u'chair', u'teddy bear', u'cell phone', u'potted plant', u'tie'],
    'significance' : "Split 4: same number of randomly generated knowns and unknowns."},
    {'known' : [u'bus', u'laptop', u'mouse', u'car', u'knife', u'donut', u'parking meter', u'cup', u'airplane', u'toaster', u'potted plant', u'carrot', u'banana', u'baseball glove', u'motorcycle', u'handbag', u'hair drier', u'horse', u'toilet', u'tennis racket', u'keyboard', u'apple', u'sheep', u'skis', u'snowboard', u'toothbrush', u'chair', u'suitcase', u'kite', u'vase', u'remote', u'oven', u'teddy bear', u'bench', u'fire hydrant', u'bed', u'stop sign', u'bowl', u'skateboard', u'bottle', u'bird', u'dog', u'sink', u'couch', u'truck', u'traffic light', u'backpack', u'train', u'elephant', u'cell phone', u'scissors', u'cow', u'spoon', u'baseball bat', u'microwave', u'bicycle', u'dining table', u'cake', u'orange', u'zebra'],
     'unknown' : [u'bear', u'boat', u'book', u'broccoli', u'cat', u'clock', u'fork', u'frisbee', u'giraffe', u'hot dog', u'person', u'pizza', u'refrigerator', u'sandwich', u'sports ball', u'surfboard', u'tie', u'tv', u'umbrella', u'wine glass'],
    'significance' : "Split 5: more randomly generated knowns than unknowns."},
    {'known' : [u'stop sign', u'truck', u'cow', u'sports ball', u'bowl', u'carrot', u'teddy bear', u'laptop', u'hair drier', u'donut', u'scissors', u'boat', u'elephant', u'backpack', u'spoon', u'banana', u'zebra', u'book', u'frisbee', u'suitcase', u'bird', u'skis', u'bus', u'giraffe', u'fork', u'tennis racket', u'microwave', u'surfboard', u'traffic light', u'couch', u'knife', u'baseball bat', u'orange', u'umbrella', u'clock', u'baseball glove', u'sink', u'tie', u'wine glass', u'bottle', u'skateboard', u'sheep', u'horse', u'cake', u'motorcycle', u'parking meter', u'remote', u'airplane', u'cell phone', u'kite', u'cup', u'snowboard', u'bench', u'person', u'car', u'refrigerator', u'fire hydrant', u'potted plant', u'handbag', u'oven'],
     'unknown' : [u'apple', u'bear', u'bed', u'bicycle', u'broccoli', u'cat', u'chair', u'dining table', u'dog', u'hot dog', u'keyboard', u'mouse', u'pizza', u'sandwich', u'toaster', u'toilet', u'toothbrush', u'train', u'tv', u'vase'],
    'significance' : "Split 6: more randomly generated knowns than unknowns." 
},
    {'known' : [u'cell phone', u'frisbee', u'fire hydrant', u'toaster', u'apple', u'person', u'bicycle', u'toothbrush', u'donut', u'broccoli', u'cat', u'toilet', u'airplane', u'bed', u'tennis racket', u'sandwich', u'baseball bat', u'parking meter', u'kite', u'banana', u'sink', u'bottle', u'bench', u'zebra', u'remote', u'oven', u'elephant', u'knife', u'cow', u'motorcycle', u'refrigerator', u'dog', u'tie', u'scissors', u'microwave', u'sheep', u'vase', u'snowboard', u'cake', u'backpack', u'truck', u'horse', u'skis', u'umbrella', u'sports ball', u'suitcase', u'bird', u'baseball glove', u'couch', u'car', u'skateboard', u'cup', u'handbag', u'tv', u'mouse', u'chair', u'hot dog', u'stop sign', u'hair drier', u'train', u'pizza', u'orange', u'potted plant', u'book', u'fork', u'teddy bear', u'wine glass', u'bear', u'giraffe', u'laptop', u'boat', u'bus'],
     'unknown' : [u'bowl', u'carrot', u'clock', u'dining table', u'keyboard', u'spoon', u'surfboard', u'traffic light'],
    'significance' : "Split 7: few randomly generated unknowns." 
},
    {'known' : [u'car', u'pizza', u'fork', u'clock', u'vase', u'backpack', u'dog', u'train', u'book', u'donut', u'skateboard', u'teddy bear', u'potted plant', u'oven', u'hot dog', u'bottle', u'airplane', u'broccoli', u'keyboard', u'cat', u'bench', u'sink', u'hair drier', u'umbrella', u'dining table', u'skis', u'cell phone', u'kite', u'chair', u'parking meter', u'bear', u'horse', u'elephant', u'motorcycle', u'couch', u'toilet', u'tennis racket', u'refrigerator', u'microwave', u'cake', u'spoon', u'tie', u'baseball glove', u'banana', u'zebra', u'cup', u'person', u'carrot', u'suitcase', u'remote', u'truck', u'toothbrush', u'knife', u'cow', u'scissors', u'bird', u'bus', u'laptop', u'snowboard', u'frisbee', u'surfboard', u'sandwich', u'bowl', u'toaster', u'mouse', u'stop sign', u'fire hydrant', u'baseball bat', u'boat', u'handbag', u'tv', u'wine glass'],
     'unknown' : [u'apple', u'bed', u'bicycle', u'giraffe', u'orange', u'sheep', u'sports ball', u'traffic light'],
    'significance' : "Split 8: few randomly generated unknowns." 
},
    {'known' : ['person','spoon','broccoli','knife','dining table','tv','book','fork','chair','clock','potted plant','oven','pizza','sink','bowl','toaster'],
     'unknown' : ['banana','orange','wine glass','refrigerator','remote','laptop','toilet','hair drier','microwave','scissors','bird','motorcycle','car','horse'],
    'significance' : "Split 9: knowing only indoor classes, try to discover mostly other indoor classes, with a few outdoor classes for control."}]
