import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np


class BLSTMWithIvector(torch.nn.Module):
    '''
        Bidirectional LSTM model with i-vectors
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--blstm-hdim', type=int, default=512)
        parser.add_argument('--blstm-num-layers', type=int, default=4)
        parser.add_argument('--blstm-dropout', type=float, default=0.1)
        parser.add_argument('--blstm-prefinal-dim', type=int, default=256)
        parser.add_argument('--ivector-layers', nargs='*', type=int, default=[1])
         
    @classmethod
    def build_model(cls, conf):
        model = BLSTMWithIvector(
            conf['idim'], conf['num_targets'], conf['ivector_dim'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            ivector_layers=conf['ivector_layers'],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],
            batch_norm_dropout=True
        )   
        return model
    
    def __init__(
        self, idim, odim, ivector_dim,
        odims=[512, 512, 512, 512, 512, 512],
        ivector_layers=[1],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1, batch_norm_dropout=True
    ):
        super().__init__()
        
        # Proper BLSTM layers
        self.batch_norm_dropout = batch_norm_dropout
        self.dropout = dropout
        self.nonlin = nonlin
        self.subsample = subsample
        self.ivector_layers = sorted(ivector_layers)

        # Add 1 feed-forward layer for every layer we need
        # to input the i-vector
        self.ivector_nn = torch.nn.ModuleList()
        for layer in self.ivector_layers:
            if layer > len(odims) - 1:
                break
            out_dim = idim if layer == 0 else odims[layer-1]
            self.ivector_nn.append(
                torch.nn.Linear(
                    ivector_dim, out_dim
                )
            )
        
        self.blstm = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        
        next_input_dim = idim
        for cur_odim in odims:
            self.blstm.append(
                torch.nn.LSTM(
                    next_input_dim, cur_odim//2, 1,
                    batch_first=True, bidirectional=True
                )
            )
            self.batchnorm.append(
                torch.nn.BatchNorm1d(cur_odim, eps=1e-03, affine=False)
            )
            next_input_dim = cur_odim

        # Last few layers
        self.prefinal_affine = torch.nn.Linear(
            next_input_dim, prefinal_affine_dim,
        )
        self.batchnorm.append(
            torch.nn.BatchNorm1d(
                prefinal_affine_dim, eps=1e-03, affine=False
            )
        )
        self.final_affine = torch.nn.Linear(
            prefinal_affine_dim, odim,
        )

    def forward(self, sample):
        xs_pad = sample.input
        ivec_pad = sample.ivector
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
       
        # Basic pattern is (blstm, relu, batchnorm, dropout) x num_layers
        # We use ivectors at the specified layers to scale the inputs,
        # which is the previous layer's output activations

        ivec_nn_layer = 0
        # Now iterate over BLSTM layers
        for i, (blstm, batchnorm) in enumerate(zip(self.blstm, self.batchnorm[:-1])):

            # Use i-vector if it is in the config
            if i in self.ivector_layers:
                ivec_transformed = self.nonlin(self.ivector_nn[ivec_nn_layer](ivec_pad))
                xs_pad = xs_pad * ivec_transformed.unsqueeze(1)
                ivec_nn_layer += 1
            
            xs_pad = blstm(xs_pad)[0]
            xs_pad = self.nonlin(xs_pad)
            if not self.batch_norm_dropout: 
                xs_pad = batchnorm(xs_pad.transpose(0,1)).transpose(0,1)
                xs_pad = F.dropout(xs_pad, p=self.dropout, training=self.training)
      
        # A few final layers
        end_idx = xs_pad.size(1) if right_context == 0 else -right_context
        output2 = xs_pad[:, left_context:end_idx:self.subsample, :]
        xs_pad = self.nonlin(self.prefinal_affine(xs_pad))
        if not self.batch_norm_dropout:
            xs_pad = self.batchnorm[-1](xs_pad)
        
        # This is basically just glue
        output = self.final_affine(xs_pad)
        return (
            output[:, left_context:end_idx:self.subsample, :],
            output2,
        )


class ChainBLSTMWithIvector(BLSTMWithIvector):
    @classmethod
    def build_model(cls, conf):
        model = ChainBLSTMWithIvector(
            conf['idim'], conf['num_targets'], conf['ivector_dim'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            ivector_layers=conf['ivector_layers'],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],
            batch_norm_dropout=True
        )   
        return model

    def __init__(
        self, idim, odim, ivector_dim,
        odims=[512, 512, 512, 512, 512, 512],
        ivector_layers=[1],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1, batch_norm_dropout=True
    ):
        super().__init__(
            idim, odim, ivector_dim, odims, ivector_layers, prefinal_affine_dim,
            nonlin, dropout, subsample
        )
        self.prefinal_xent = torch.nn.Linear(
            odims[-1],
            prefinal_affine_dim,
        )
        self.xent_batchnorm = torch.nn.BatchNorm1d(
            prefinal_affine_dim,
            eps=1e-03, affine=False
        )
        self.xent_layer = torch.nn.Linear(prefinal_affine_dim, odim)
    
    def forward(self, xs_pad):
        output, xs_pad = super().forward(xs_pad)
        if self.training:
            xs_pad = self.nonlin(self.prefinal_xent(xs_pad))
            if not self.batch_norm_dropout:
                xs_pad = self.xent_batchnorm(xs_pad)
            xs_pad = self.xent_layer(xs_pad)
        return output, xs_pad 

