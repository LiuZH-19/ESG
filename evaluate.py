import numpy as np
import torch
from collections import defaultdict


def get_scores(output:np.ndarray, groud_truth:np.ndarray, mask, out_catagory: str, detail= False):
    """
    evluate the model performance
    :param output: [n_samples, 12, n_nodes, n_features]
    :param groud_truth: [n_samples, 12, n_nodes, n_features]
    :return: dict [str -> float]
    """
    if  torch.is_tensor(output):
        output = output.cpu().numpy()
        groud_truth = groud_truth.cpu().numpy()
    if out_catagory == 'multi':
        if mask:
            if output.shape != groud_truth.shape:
                groud_truth = np.expand_dims( groud_truth[...,0], axis=-1)
            assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
            batch, steps, scores = output.shape[0], output.shape[1], defaultdict(dict)
            if detail:
                for step in range(steps):
                    y_pred = np.reshape(output[:,step],(batch, -1))
                    y_true = np.reshape(groud_truth[:,step],(batch,-1))
                    scores['MAE'][f'horizon-{step}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
                    scores['RMSE'][f'horizon-{step}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
                    scores['MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
            scores['MAE']['all'] = masked_mae_np(output,groud_truth ,null_val=0.0)
            scores['RMSE']['all'] = masked_rmse_np( output,groud_truth, null_val=0.0)
            scores['MAPE']['all'] = masked_mape_np( output,groud_truth, null_val=0.0) * 100.0
        else:
            if output.shape != groud_truth.shape:
                groud_truth = np.expand_dims( groud_truth[...,0], axis=-1)
            assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
            batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
            if detail:
                for step in range(steps):
                    y_pred = output[:,step]
                    y_true = groud_truth[:,step]
                    scores['MAE'][f'horizon-{step}'] = mae_np(y_pred, y_true)
                    scores['RMSE'][f'horizon-{step}'] = rmse_np(y_pred, y_true)
                    # scores['MAPE'][f'horizon-{step}'] = mape_np(y_pred,y_true) * 100.0
                    scores['MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
                    scores['StemGNN_MAPE'][f'horizon-{step}'] = stemgnn_mape(y_pred, y_true) * 100.0
                    scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                    scores['CORR'][f'horizon-{step}']= node_pcc_np(y_pred.swapaxes(1,-1).reshape((-1,node)), y_true.swapaxes(1,-1).reshape((-1,node)))
            scores['MAE']['all'] = mae_np(output,groud_truth)
            scores['RMSE']['all'] = rmse_np(output,groud_truth)
            scores['MAPE']['all'] = masked_mape_np( output,groud_truth, null_val=0.0) * 100.0
            scores['StemGNN_MAPE']['all'] = stemgnn_mape(output,groud_truth) * 100.0
            scores['PCC']['all'] = pcc_np(output,groud_truth)
            scores['CORR']['all'] = node_pcc_np(output.swapaxes(2,-1).reshape((-1,node)), groud_truth.swapaxes(2,-1).reshape((-1,node)))
    else:
        output = output.squeeze()
        groud_truth = groud_truth.squeeze()
        assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
        scores = defaultdict(dict)

        scores['RMSE'] = rmse_np(output, groud_truth)
        scores['masked_MAPE']= masked_mape_np(output, groud_truth, null_val=0.0) * 100.0
        scores['CORR']= node_pcc_np(output, groud_truth)
        scores['RSE'] = rse_np(output, groud_truth)
        scores['MAPE2'] = stemgnn_mape(output, groud_truth) * 100.0
        scores['MAE'] = mae_np(output, groud_truth)
    return scores

def rse_np(preds, labels):
    mse = np.sum(np.square(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.square(np.subtract(labels, means)).astype('float32'))
    return np.sqrt(mse/labels_mse)
    
def node_pcc_np(x, y):
    sigma_x = x.std(axis=0)
    sigma_y = y.std(axis=0)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    cor = ((x - mean_x) * (y - mean_y)).mean(0) / (sigma_x * sigma_y + 0.000000000001)
    return cor.mean()

def mae_np(preds, labels):
    mae = np.abs(np.subtract(preds, labels)).astype('float32')
    return np.mean(mae)


def rmse_np(preds, labels):
    mse = mse_np(preds, labels)
    return np.sqrt(mse)

def mse_np(preds, labels):
    return np.mean(np.square(np.subtract(preds, labels)).astype('float32'))

def mape_np(preds, labels):
    mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    return np.mean(mape)

def pcc_np(x, y):
    x,y = x.reshape(-1),y.reshape(-1)
    return np.corrcoef(x,y)[0][1]


def stemgnn_mape(preds,labels, axis=None):
    '''
    Mean absolute percentage error.
    :param labels: np.ndarray or int, ground truth.
    :param preds: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(preds - labels) / (np.abs(labels)+1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)