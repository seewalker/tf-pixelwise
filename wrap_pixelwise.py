import argparse
from pixelwise import *

parser = argparse.ArgumentParser()
parser.add_argument('action')
parser.add_argument('dataset')
parser.add_argument('splitid')
parser.add_argument('nickname')
parser.add_argument('-batchsize',default=42)
parser.add_argument('-trestart',default=0)
parser.add_argument('-anticipate_missing',default=1)
parser.add_argument('-biased_upsampling',default=0)
parser.add_argument('-numfuse',default=2)
parser.add_argument('-device',default="GPU")
parser.add_argument('-savestep',default=20)
parser.add_argument('-restart',default=True)
parser.add_argument('-loadstep',default="max")
parser.add_argument('-imgdir',default=None)
parser.add_argument('-trial',default=0)

args = parser.parse_args()
if args.action == 'train':
    train(args.nickname,splitid=args.splitid,numfuse=int(args.numfuse),trestart=int(args.trestart),anticipate_missing=int(args.anticipate_missing),dataset=args.dataset,device=args.device,savestep=args.savestep,use_bias=int(args.biased_upsampling),batchsize=int(args.batchsize))
elif args.action == 'test':
    test(args.nickname,splitid=args.splitid,numfuse=int(args.numfuse),trestart=int(args.trestart),anticipate_missing=int(args.anticipate_missing),dataset=args.dataset,device=args.device,savestep=args.savestep,use_bias=int(args.biased_upsampling),batchsize=int(args.batchsize))
elif args.action == 'run':
    run(args.nickname,int(args.trial),int(args.splitid),int(args.numfuse),args.imgdir)
