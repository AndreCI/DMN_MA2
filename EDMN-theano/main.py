import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json

from utils import utils
from utils import run
from utils import nn_utils



#Getting, parsing and printing input arguments.

print("==> parsing input arguments")
parser = argparse.ArgumentParser(prog="ExtDMN",description="Andre s code ExtDMN. Use for Q&A, master semester project @EPFL-LIA, 2017")

parser.add_argument('--network', type=str, default="dmn_batch", choices=['dmn_basic', 'dmn_multiple','dmn_smooth','dmn_batch'], help='network type: dmn_basic, dmn_multiple, dmn_smooth, or dmn_batch')
parser.add_argument('--word_vector_size', type=int, default=50, choices=['50','100','200','300'], help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
parser.add_argument('--answer_step_nbr',type=int,default=1, help='Number of step done by the answer module (>0)')
parser.add_argument('--mode', type=str, default="train", help='mode: train, test or minitest. Test and minitest mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=5, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.set_defaults(shuffle=True)
args = parser.parse_args()

print(args)


#Checking if the vector size is valid (using GloVe here, so no fancy size allowed)
assert args.word_vector_size in [50, 100, 200, 300]

#Getting the network_name and parameters
#TODO: should probably recode this, this isn't only the network name but much more
network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.babi%s' % (
    args.network, 
    args.memory_hops, 
    args.dim, 
    args.batch_size, 
    ".na" if args.normalize_attention else "", 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "",
    args.babi_id)

#Getting dataset(train & test)
if args.network == 'dmn_multiple':
    babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_test_id, multiple=True)
else:
    babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_test_id)

#Getting GloVe, i.e. embedding matrix
word2vec = utils.load_glove(args.word_vector_size)

#Wrapping everything into args_dict
args_dict = dict(args._get_kwargs())
args_dict['babi_train_raw'] = babi_train_raw
args_dict['babi_test_raw'] = babi_test_raw
args_dict['word2vec'] = word2vec
    

# init class, choose network depending on arguments
if args.network == 'dmn_batch':
    from models import dmn_batch
    dmn = dmn_batch.DMN_batch(**args_dict)

elif args.network == 'dmn_multiple':
    from models import dmn_multiple
    if (args.batch_size != 1):
        print("==> no minibatch training, argument batch_size is useless")
        args.batch_size = 1
    dmn = dmn_multiple.DMN_multiple(**args_dict)

elif args.network == 'dmn_basic':
    from models import dmn_basic
    if (args.batch_size != 1):
        print("==> no minibatch training, argument batch_size is useless")
        args.batch_size = 1
    dmn = dmn_basic.DMN_basic(**args_dict)

elif args.network == 'dmn_smooth':
    from models import dmn_smooth
    if (args.batch_size != 1):
        print("==> no minibatch training, argument batch_size is useless")
        args.batch_size = 1
    dmn = dmn_smooth.DMN_smooth(**args_dict)

elif args.network == 'dmn_qa':
    from models import dmn_qa_draft
    if (args.batch_size != 1):
        print("==> no minibatch training, argument batch_size is useless")
        args.batch_size = 1
    dmn = dmn_qa_draft.DMN_qa(**args_dict)

else: 
    raise Exception("No such network known: " + args.network)
    

#Try to load a pretrained network
if args.load_state != "":
    dmn.load_state(args.load_state)
    

if args.mode == 'train':
    print("==> training")
    skipped = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if args.shuffle:
            dmn.shuffle_train_set()
        
        _, skipped = run.do_epoch(args, dmn,'train', epoch, skipped)
        
        epoch_loss, skipped = run.do_epoch(args, dmn, 'test', epoch, skipped)
        
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)

        if (epoch % args.save_every == 0):    
            print("==> saving ... %s" % state_name)
            dmn.save_params(state_name, epoch)
        print("epoch %d took %.3fs" % (epoch, float(time.time()) - start_time))
        
elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    run.do_epoch('test', 0)

elif args.mode == 'minitest':
    file = open('last_tested_model.json','w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    #json.dump(data, file, indent=2)
    multiple_ans = args.network == 'dmn_multiple'
    run.get_stat(dmn, data["vocab"], nbr_stat=200)
    #run.do_minitest(dmn, data["vocab"], multiple_ans=multiple_ans, nbr_test=5)#, log_it=True, state_name=args.load_state)
    
else:
    raise Exception("unknown mode")
