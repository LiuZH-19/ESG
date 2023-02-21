# Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting

This is PyTorch implementation of ESG in the following paper:

[Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting](https://dl.acm.org/doi/abs/10.1145/3534678.3539274)

## Requirements

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
cd data

# NYC-Bike
python gen_npz.py --h5_name nyc-bike

# NYC-Taxi
python gen_npz.py --h5_name nyc-taxi
```

## Model Training

### Single-step

- Solar-Energy  

```
python train_single_step.py  --data solar-energy --expid <expid>  --num_nodes 137 --batch_size 16 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

- Electricity

```
python train_single_step.py  --data electricity --expid <expid>  --num_nodes 321 --batch_size 4 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

- Exchange Rate 

```
python train_single_step.py  --data exchange-rate --expid <expid>  --num_nodes 8 --batch_size 4 --dy_embedding_dim 16 --runs 10 --horizon 3 
```

- Wind

```
python train_single_step.py  --data wind --expid <expid>  --num_nodes 28 --batch_size 32 --dy_embedding_dim 20 --runs 10 --horizon 3 
```

### Multi-step

- NYC-Bike

```
python train_multi_step.py --data nyc-bike --expid <expid> --num_nodes 250 --batch_size 16  --dy_embedding_dim 20 --runs 10
```

- NYC-Taxi

```
python train_multi_step.py --data nyc-taxi --expid <expid> --num_nodes 266 --batch_size 16  --dy_embedding_dim 20 --runs 10
```

## Run the trained Model

You can run the following command to evaluate the test datasets using the trained model.

```
#Solar-Energy
python test_single.py  --data solar-energy --horizon 24 --expid only_test

#Electricity
python test_single.py  --data electricity --horizon 24 --expid only_test

#Exchange Rate
python test_single.py  --data exchange-rate --horizon 24 --expid only_test

#wind
python test_single.py  --data wind --horizon 24 --expid only_test

#NYC-Bike
python test_multi.py  --data nyc-bike --expid only_test

#NYC-Taxi
python test_multi.py  --data nyc-taxi --expid only_test
 
```

## Acknowledgements

The implementation of ESG relies on resources from the following  repository, we thank the original authors for open-sourcing their work. 

https://github.com/nnzhan/MTGNN

## Citation

Please consider citing if you find this code useful to your research.

```
@inproceedings{ye2022esg,
author = {Ye, Junchen and Liu, Zihan and Du, Bowen and Sun, Leilei and Li, Weimiao and Fu, Yanjie and Xiong, Hui},
title = {Learning the Evolutionary and Multi-Scale Graph Structure for Multivariate Time Series Forecasting},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539274},
doi = {10.1145/3534678.3539274},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {2296–2306},
numpages = {11},
location = {Washington DC, USA},
series = {KDD '22}
}
```



