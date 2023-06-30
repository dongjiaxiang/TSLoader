

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *
import random

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'ts_mergedatasets',
        ]

class DatasetCatalog:
    DATASETS = {
        'etth1': {
            "root_path": "/storage/dataset/DSET924f39a246e2bcba76feef284556"
        },
        'etth2': {
            "root_path": "/storage/dataset/DSETecc2e54a4c80a793255c932e7b72"
        },
        'ettm1': {
            "root_path": "/storage/dataset/DSET778fcf74414d8e186fd05350ebee"
        },
        'ettm2': {
            "root_path": "/storage/dataset/DSETe9eb0a5a4b40876add2dbd3acb6a"
        },
        'electricity': {
            "root_path": "/storage/dataset/DSET73e1e542467986886113370b39d1"
        },
        'traffic': {
            "root_path": "/storage/dataset/DSET69dd739245f59853a74d98d2cc4c"
        },
        'weather': {
            "root_path": "/storage/dataset/DSETb990ae96465d9eff1bfff43e5eca"
        },
        'illness': {
            "root_path": "/storage/dataset/DSET8a91cb7146f58a081f7fe7561dea"
        },
        'exchange': {
            "root_path": "/storage/dataset/DSETffd84b7f4e4e81ad73db993d91e8"
        },
        'm4': {
            "root_path": "/storage/dataset/DSET14960c3e4f4f8455ea397c95d6fc/m4"
        },
        'ts_mergedatasets': {
            "root_path": "/storage/dataset/DSET9bcbf88c493b9581586dfe1bfbc8"
        }
    }

    @staticmethod
    def get(name):
        if name in DatasetCatalog.DATASETS:
            return DatasetCatalog.DATASETS[name]

        raise RuntimeError("Dataset not available: {}".format(name))

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    Data_ATTRS = DatasetCatalog.get(params.dset)
    root_path = Data_ATTRS['root_path']

    if params.dset == 'ettm1':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    

    elif params.dset == 'electricity':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'ts_mergedatasets':
        size = [params.context_points, 0, params.target_points]
        random_seed = random.randint(1, 100)
        dls = DataLoaders(
                datasetCls=Dataset_TSMerge,
                dataset_kwargs={
                'root_path': root_path,
                'size': size,
                'random_seed': random_seed
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    print(dls.vars, dls.len)
    #dls.c = dls.train.dataset[0][1].shape[0]

    return dls



#if __name__ == "__main__":
#    class Params:
#        dset= 'etth2'
#        context_points= 384
#        target_points= 96
##        batch_size= 64
 #       num_workers= 8
 #       with_ray= False
 #       features='M'
 #   params = Params
 #   dls = get_dls(params)
 #   for i, batch in enumerate(dls.valid):
 #       print(i, len(batch), batch[0].shape, batch[1].shape)
 #   breakpoint()
