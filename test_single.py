from os import write
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn

from model.esg import ESG
from util import *
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




parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data', type=str, default='electricity', help='the name of the dataset')
parser.add_argument('--horizon', type=int, default=24)
parser.add_argument('--expid', type=str, default='1',
                    help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--seq_in_len',type=int,default=168,help='input sequence length') 
parser.add_argument('--batch_size',type=int,default=4, help='batch size')
parser.add_argument('--runs',type=int,default=10,help='number of runs')



args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

def main():

    save_folder = os.path.join('saves', args.data, args.expid, 'horizon_'+str(args.horizon))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join('trained',args.data+'_'+str(args.horizon)+'.pt')

    sys.stdout = Logger(os.path.join(save_folder,'log.txt')) 

    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize) 

    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)
    

    
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)
    
    #model = model.to(device)

    val_scores, P, Y = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)
    test_scores, predict, Ytest = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size)

    print("valid rse {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f}".format(val_scores['RSE'], val_scores['RAE'], val_scores['CORR']))
    
    
    # save test results
    np.savez(os.path.join(save_folder,'test-results.npz'), predictions=predict, targets=Ytest)
    print(json.dumps(test_scores, cls=JsonEncoder, indent=4))
   
    with open(os.path.join(save_folder,'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=JsonEncoder, indent=4)
    
    # print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_scores['RSE'], test_scores['RAE'], test_scores['CORR']))
   
   
    
if __name__ == "__main__":
    main()
