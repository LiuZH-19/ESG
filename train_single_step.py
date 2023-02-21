from os import write
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from model.esg import ESG
from util import *
from trainer import Optim
from evaluate import get_scores

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
    

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

#log
class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass



def evaluate(data, inputs, targets, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(inputs, targets, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3) #(B, F, N, T)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rae = (total_loss_l1 / n_samples) / data.rae

    scale = data.scale.expand(predict.size(0), data.m)
    predict = predict * scale
    test = test * scale
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    print("predict:")
    print(predict.shape)
    print("Ytest")
    print(Ytest.shape) 
    scores = get_scores(predict, Ytest,0, 'single')
    scores['RAE'] = rae.item()     
    return scores, predict, Ytest


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0   
    n_samples = 0
    iter = 0
    for tx, ty in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        tx = torch.unsqueeze(tx,dim=1)
        tx = tx.transpose(2,3)
                     
        output = model(tx)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, ty * scale)                  
        loss.backward()
        total_loss += loss.item()   
        n_samples += (output.size(0) * data.m)
        grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.8f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data', type=str, default='solar-energy', help='the name of the dataset')
parser.add_argument('--horizon', type=int, default=24)
# parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
#                     help='report interval')
parser.add_argument('--expid', type=str, default='1',
                    help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)

parser.add_argument('--dy_embedding_dim',type=int,default=20, help='the dimension of evolving node representation')
parser.add_argument('--dy_interval',type=list, default=[31,31,21,14,1],help='time intervals for each layer')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--seq_in_len',type=int,default=168,help='input sequence length') 
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--kernel_set',type=list,default=[2,3,6,7],help='the kernel set in TCN')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
# parser.add_argument('--fc_dim',type=int,default=504288,help='fc_dim') #so 504288 el  252224 ex 72560   wind 104896
parser.add_argument('--st_embedding_dim',type=int,default=40,help='the dimension of static node representation')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')

parser.add_argument('--batch_size',type=int,default=16, help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--early_stop',type=str_to_bool,default=True,help='')
parser.add_argument('--early_stop_steps',type=int,default=15,help='')
parser.add_argument('--runs',type=int,default=10,help='number of runs')



args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main(runid):

    save_folder = os.path.join('saves', args.data, args.expid, 'horizon_'+str(args.horizon), str(runid))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder,'best-model.pt')

    sys.stdout = Logger(os.path.join(save_folder,'log.txt')) 

    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize) 

    
    node_fea = get_node_fea(args.data, 0.6)
    node_fea = torch.tensor(node_fea).type(torch.FloatTensor).to(args.device)
    model = ESG(args.dy_embedding_dim, args.dy_interval, args.num_nodes, args.seq_in_len, args.seq_out_len, args.in_dim, args.out_dim, 1, args.layers,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels, kernel_set=args.kernel_set,
                  dilation_exp=args.dilation_exponential, gcn_depth=args.gcn_depth,
                  device=device, fc_dim = (node_fea.shape[0]-18)*16, st_embedding_dim=args.st_embedding_dim,                  
                  dropout=args.dropout,  propalpha=args.propalpha, layer_norm_affline=False,
                  static_feat = node_fea)
 
    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    print(model)

    run_folder = os.path.join(save_folder,'run')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    writer = SummaryWriter(run_folder)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)
    

    best_val = 10000000
    best_epoch = 10000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )
        
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1, 1):
            epoch_start_time = time.time()
            train_loss= train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            writer.add_scalars('train_loss', {'train':train_loss}, epoch)
            val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_scores['RSE'], val_scores['RAE'], val_scores['CORR']), flush=True)
            writer.add_scalars('rse',{'valid':val_scores['RSE']},global_step = epoch)
            writer.add_scalars('corr',{'valid':val_scores['CORR']},global_step = epoch)
           
            # Save the model if the validation loss is the best we've seen so far.
            val_loss = val_scores['RSE']
            if val_loss < best_val:
                print('save the model at epoch ' + str(epoch)+'*********')
                with open( model_path, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
                best_epoch = epoch
            elif args.early_stop and  epoch - best_epoch > args.early_stop_steps:
                print('best epoch:', best_epoch)
                raise ValueError('Early stopped.')
            if epoch % 5 == 0:
                test_scores, P, Y = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size)
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format( test_scores['RSE'], test_scores['RAE'], test_scores['CORR']), flush=True)
                writer.add_scalars('rse',{'test':test_scores['RSE']},global_step = epoch)
                writer.add_scalars('corr',{'test':test_scores['CORR']},global_step = epoch)
    
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        # except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    
    print('best epoch:', best_epoch)   
    
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f)

    val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_scores, predict, Ytest = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    
    # save test results
    np.savez(os.path.join(save_folder,'test-results.npz'), predictions=predict, targets=Ytest)
    print(json.dumps(test_scores, cls=JsonEncoder, indent=4))
   
    with open(os.path.join(save_folder,'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=JsonEncoder, indent=4)
    
    #print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_scores['RSE'], test_scores['RAE'], test_scores['CORR']))
    return val_scores['RSE'], val_scores['RAE'], val_scores['CORR'], test_scores['RSE'], test_scores['RAE'], test_scores['CORR']
   
    
if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    for i in range(args.runs):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
    print('\n\n')
    print(str(args.runs)+' runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
    print('\n\n')
    print("test\trse\trae\tcorr")
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))

