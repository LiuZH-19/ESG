import torch
import numpy as np
import argparse
from util import *

from model.esg import ESG
import sys,os
from evaluate import get_scores


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

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--data',type=str,default='nyc-bike',help='the name of the dataset')
parser.add_argument('--expid',type=str,default='1',help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--mask0',type=str_to_bool,default=False,help='whether to mask value 0 in the raw data')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')



args = parser.parse_args()
torch.set_num_threads(3)

def main():
    #load data
    save_folder = os.path.join('./saves',args.data, args.expid)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join('trained',args.data+'.pt')

    sys.stdout = Logger(os.path.join(save_folder,'log.txt')) 
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']


    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)


    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:] #[B, C, N,T]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds =model(testx).transpose(1,3)  #[B,T,N,1] # engine.model
        outputs.append(preds)#[B,C,N,T]

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    preds = scaler.inverse_transform(yhat).transpose(1,-1).cpu().numpy()
    targets = realy.transpose(1,-1).cpu().numpy()
    valid_scores = get_scores(preds, targets, mask = args.mask0, out_catagory='multi', detail = True)

    print('valid results:')
    print(json.dumps(valid_scores, cls=MyEncoder, indent=4))

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device) #[B, T, N,C]
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:] #[B, C, N,T]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)  #[B,T,N,1] engine.
            preds = preds.transpose(1, 3) 
        outputs.append(preds)#[B,C,N,T]

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    save_predicitions = scaler.inverse_transform(yhat).transpose(1,-1).cpu().numpy()
    save_targets = realy.transpose(1,-1).cpu().numpy()    
    test_scores = get_scores(save_predicitions, save_targets, mask = args.mask0, out_catagory='multi', detail = True)
   
    print('test results:')
    print(json.dumps(test_scores, cls=MyEncoder, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=MyEncoder, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=save_predicitions, targets=save_targets)



if __name__ == "__main__":
    main()




