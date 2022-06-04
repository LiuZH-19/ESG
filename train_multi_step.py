import torch
import numpy as np
import argparse
import time
from tensorboardX import SummaryWriter
from util import *
from trainer import Trainer

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

parser.add_argument('--dy_embedding_dim',type=int,default=20,help='the dimension of evolving node representation')
parser.add_argument('--dy_interval',type=list, default=[1,1,1],help='time intervals for each layer')
parser.add_argument('--num_nodes',type=int,default=250,help='number of nodes/variables')#bike 250 taxi 266
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')
parser.add_argument('--kernel_set',type=list,default=[2,6],help='the kernel set in TCN')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--fc_dim',type=int,default= 95744,help='fc_dim') #bike 95744 taxi 95744
parser.add_argument('--st_embedding_dim',type=int,default=40,help='the dimension of static node representation')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')

parser.add_argument('--mask0',type=str_to_bool,default=False,help='whether to mask value 0 in the raw data')

parser.add_argument('--cl', type=str_to_bool, default=False,help='whether to do curriculum learning')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=300,help='')
parser.add_argument('--early_stop',type=str_to_bool,default=True,help='')
parser.add_argument('--early_stop_steps',type=int,default=30,help='')
parser.add_argument('--print_interval',type=int,default=50,help='')
parser.add_argument('--runs',type=int,default=10,help='number of runs')



args = parser.parse_args()
torch.set_num_threads(3)

def main(runid):
    #load data
    save_folder = os.path.join('./saves',args.data, args.expid, str(runid))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder,'best-model.pt')
    sys.stdout = Logger(os.path.join(save_folder,'log.txt')) 
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    node_fea = get_node_fea(args.data, 0.7)
    node_fea = torch.tensor(node_fea).type(torch.FloatTensor).to(args.device)
   
    model = ESG(args.dy_embedding_dim, args.dy_interval, args.num_nodes, args.seq_in_len, args.seq_out_len, args.in_dim, args.out_dim, 1, args.layers,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels, kernel_set=args.kernel_set,
                  dilation_exp=args.dilation_exponential, gcn_depth=args.gcn_depth,
                  device=device, fc_dim = args.fc_dim, st_embedding_dim=args.st_embedding_dim,                  
                  dropout=args.dropout,  propalpha=args.propalpha, layer_norm_affline=False,
                  static_feat = node_fea)
  
    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    print(model)
    run_folder = os.path.join(save_folder,'run')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    writer = SummaryWriter(run_folder)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl,args.mask0)
    
    try:
        print("start training...",flush=True)
        his_loss =[]
        val_time = []
        train_time = []
        minl = 1e5
        best_epoch = 1e5
        for i in range(1,args.epochs+1):
            train_loss = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                tx= trainx.transpose(1, 3) #[B,C,N,T]
                trainy = torch.Tensor(y).to(device)
                ty = trainy.transpose(1, 3)
                   
                loss = engine.train(tx, ty[:,:args.out_dim,:,:])
                train_loss.append(loss)

                if iter % args.print_interval == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}'
                    print(log.format(iter, train_loss[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
           

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)

                vloss = engine.eval(testx, testy[:,:args.out_dim,:,:])
                valid_loss.append(vloss)
               
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)

            mvalid_loss = np.mean(valid_loss)
            his_loss.append(mvalid_loss)

           
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mvalid_loss, (t2 - t1)),flush=True)
            writer.add_scalars('loss', {'train':mtrain_loss}, global_step=i )
            writer.add_scalars('loss', {'valid':mvalid_loss}, global_step=i )
           

            if mvalid_loss<minl:
                with open(model_path, 'wb') as f:
                    torch.save(engine.model, f)
                #torch.save(engine.model.state_dict(), os.path.join(save_folder,"exp.pth"))
                minl = mvalid_loss
                best_epoch = i
                print(f'save best epoch {best_epoch} *****************')
            elif args.early_stop and  i - best_epoch > args.early_stop_steps:
                print('best epoch:', best_epoch)
                raise ValueError('Early stopped.')
    
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        print('-' * 89)
        print('Exiting from training early')
    
    
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    

    bestid = np.argmin(his_loss)
    
    # engine.model.load_state_dict(torch.load(os.path.join(save_folder,"exp.pth")))

    # Load the best saved model.
    with open(model_path, 'rb') as f:
        engine.model = torch.load(f)

    print("Training finished")
    print('Best epoch:', bestid)
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:] #[B, C, N,T]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)  #[B,T,N,1]
            preds = preds.transpose(1,3)
        outputs.append(preds)#[B,C,N,T]

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]



    # mask = 1 
    # if args.data == 'nyc-bike' or args.data == 'nyc-taxi':
    #     mask = 0
    preds = scaler.inverse_transform(yhat).transpose(1,-1).cpu().numpy()
    targets = realy.transpose(1,-1).cpu().numpy()
    valid_scores = get_scores(preds, targets, mask = args.mask0, out_catagory='multi')
    vrmse = valid_scores['RMSE']['all']
    vmae = valid_scores['MAE']['all']
    vcorr = valid_scores['CORR']['all']
    

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device) #[B, T, N,C]
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:] #[B, C, N,T]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)  #[B,T,N,1]
            preds = preds.transpose(1, 3) 
        outputs.append(preds)#[B,C,N,T]

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    save_predicitions = scaler.inverse_transform(yhat).transpose(1,-1).cpu().numpy()
    save_targets = realy.transpose(1,-1).cpu().numpy()    
    test_scores = get_scores(save_predicitions, save_targets, mask = args.mask0, out_catagory='multi', detail = True)
    rmse = []
    mae = []
    corr = []    
    for i in range(args.seq_out_len):
        rmse.append(test_scores['RMSE'][f'horizon-{i}'])
        mae.append(test_scores['MAE'][f'horizon-{i}'])
        corr.append(test_scores['CORR'][f'horizon-{i}'])
    armse = test_scores['RMSE']['all']    
    amae = test_scores['MAE']['all']
    acorr = test_scores['CORR']['all']
    
    print('test results:')
    print(json.dumps(test_scores, cls=MyEncoder, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=MyEncoder, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=save_predicitions, targets=save_targets)
    return vrmse, vmae, vcorr, rmse, mae, corr, armse, amae, acorr


if __name__ == "__main__":
    vrmse= []
    vmae = []
    vcorr = []
    rmse = []
    mae = []
    corr = []
    armse = []
    amae = []
    acorr = []
    for i in range(args.runs):
        v1, v2, v3, t1, t2, t3, a1, a2, a3 = main(i)
        vrmse.append(v1)
        vmae.append(v2)
        vcorr.append(v3)
        rmse.append(t1)
        mae.append(t2)
        corr.append(t3)
        armse.append(a1)
        amae.append(a2)
        acorr.append(a3)

    rmse = np.array(rmse)
    mae = np.array(mae)
    corr = np.array(corr)
    
    mrmse = np.mean(rmse,0)
    mmae = np.mean(mae,0)
    mcorr = np.mean(corr,0)  

    srmse = np.std(rmse,0)
    smae = np.std(mae,0)
    scorr = np.std(corr,0)
    

    print(f'\n\nResults for {args.runs} runs\n\n')
    #valid data
    print('valid\tRMSE\tMAE\tCORR')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vrmse),np.mean(vmae),np.mean(vcorr)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vrmse),np.std(vmae),np.std(vcorr)))
    print('\n\n')
    #test data
    print('test|horizon\tRMSE-mean\tMAE-mean\tCORR-mean\tRMSE-std\tMAE-std\tcorr-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, mrmse[i], mmae[i], mcorr[i], srmse[i], smae[i], scorr[i]))
    print('test|All\tRMSE-mean\tMAE-mean\tCORR-mean\tRMSE-std\tMAE-std\tcorr-std')
    print(log.format(0, np.mean(armse,0), np.mean(amae,0), np.mean(acorr,0), np.std(armse,0), np.std(amae,0), np.std(acorr,0)))





