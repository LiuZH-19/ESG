# Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting

This is PyTorch implementation of ESG in the following paper:

Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting.

## Requirements

- scipy
- torch
- tqdm
- h5py
- numpy
- pandas
- PyYAML
- tensorboardX
- torch

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

## Data Preparation

### Datasets for Single-Step Forecasting

Solar-Energy, Electricity, Exchange-rate datasets can be obtained from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them , turn them into  `*.h5` files  and move them to the  `data/h5data` folder.

Wind dataset can be obtained from https://www.kaggle.com/sohier/30-years-of-european-wind-generation.   Delete the `CY` column (All values are 0.) , and sum up the values of 24 hours per day to obtain the daily  estimates of an area’s energy potential. 

For convenience, you can  directly download the `*.h5`  files of  above datasets from the [Google Drive](https://drive.google.com/drive/folders/1LKu_fLzr4_DdusZ-IjzQbgYPAypfuJVA?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1X5c0OHWyAtwdcOU5QAzAhA?pwd=6ti1 ), and move the `h5data` folder  into the  `data` folder.

### Datasets for Multi-Step Forecasting

Download  NYC-Bike and NYC-Taxi datasets from [Google Drive](https://drive.google.com/drive/folders/1LKu_fLzr4_DdusZ-IjzQbgYPAypfuJVA?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1X5c0OHWyAtwdcOU5QAzAhA?pwd=6ti1 ) .  Move  `*.h5` files  into the `data/h5data/` folder and  put `nyc-bike` and `nyc-taxi` folders which contain train/test/val datasets into the  `data/` folder. 

You can also run the following commands to generate train/test/val dataset at `data/{nyc-bike,nyc-taxi}/{train,val,test}.npz` by using  `*.h5` files.

```
# Create data directories
mkdir -p data/{nyc-bike,nyc-taxi}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Model Training

### Single-step

- Solar-Energy  (在跑 test62)

```
python train_single_step.py  --data solar-energy --expid expid  --num_nodes 137 --fc_dim 504288 --batch_size 16 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

- Electricity

```
python train_single_step.py  --data electricity --expid expid  --num_nodes 321 --fc_dim 252224 --batch_size 4 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

- Exchange Rate (在跑 test62)

```
python train_single_step.py  --data exchange-rate --expid expid  --num_nodes 8 --fc_dim 72560 --batch_size 4 --dy_embedding_dim 16 --runs 10 --horizon 3 
```

- Wind

```
python train_single_step.py  --data wind --expid expid  --num_nodes 28 --fc_dim 104896 --batch_size 32 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

### Multi-step

- NYC-Bike

```
python train_multi_step.py --data nyc-bike --expid expid --num_nodes 250 --fc_dim 95744 --batch_size 16  --dy_embedding_dim 20 --runs 10
```

- NYC-Taxi

```
python train_multi_step.py --data nyc-taxi --expid expid --num_nodes 266 --fc_dim 95744 --batch_size 16  --dy_embedding_dim 20 --runs 10
```

## Acknowledgements

The implementation of ESG relies on resources from the following  repository, we thank the original authors for open-sourcing their work. 

https://github.com/nnzhan/MTGNN

## Citation

Please consider citing if you find this code useful to your research.

```

```



