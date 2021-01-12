import torch
import torch.nn.functional as F
import numpy as np

from pytorch_tdnn.tdnnf import TDNNF as TDNNFLayer


class TDNNF(torch.nn.Module):
    '''
        Kaldi TDNN style encoder implemented as convolutions
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--tdnnf-hdim', type=int, default=625)
        parser.add_argument('--tdnnf-bottleneck-dim', type=int, default=256)
        parser.add_argument('--tdnnf-num-layers', type=int, default=10)
        parser.add_argument('--tdnnf-dropout', type=float, default=0.1)
        parser.add_argument('--tdnnf-prefinal-dim', type=int, default=192)
        parser.add_argument('--tdnnf-bypass-scale', type=float, default=0.66)
     
    @classmethod
    def build_model(cls, conf):
        model = TDNNF(
            conf['idim'], conf['num_targets'],
            odims=[conf['tdnnf_hdim'] for i in range(conf['tdnnf_num_layers'])],
            bottleneck_dims=[conf['tdnnf_bottleneck_dim'] for i in range(conf['tdnnf_num_layers'])],
            dropout=conf['tdnnf_dropout'],
            prefinal_affine_dim=conf['tdnnf_prefinal_dim'],
            bypass_scale=conf['tdnnf_bypass_scale'],
            subsample=conf['subsample'],
            batch_norm_dropout=True
        )   
        return model
    
    def __init__(
        self, idim, odim,
        odims=[625, 625, 625, 625, 625, 625],
        bottleneck_dims=[256, 256, 256, 256, 256, 256],
        prefinal_affine_dim=625,
        time_strides=[1,-1,1,-1,0,1,3,-3,3,-3,3,-3],
        nonlin=F.relu, dropout=0.1, bypass_scale=0.66, subsample=1, batch_norm_dropout=True,
    ):
        super().__init__()
        
        # set random seed
        np.random.seed(0)
        
        # Proper TDNN layers
        odims_ = list(odims)
        odims_.insert(0, idim)
        self.batch_norm_dropout = batch_norm_dropout
        self.dropout = dropout
        self.bypass_scale = bypass_scale
        self.nonlin = nonlin
        self.subsample = subsample
        self.tdnnf = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        i = 0
        for layer in range(len(odims)):
            self.tdnnf.append(
                TDNNFLayer(
                    odims_[i], odims_[i+1], bottleneck_dims[i], time_strides[i]
                )
            )
            self.batchnorm.append(
                torch.nn.BatchNorm1d(odims_[i+1], eps=1e-03, affine=False)
            )
            i += 1

        # Last few layers
        self.prefinal_affine = torch.nn.Conv1d(
            odims_[i], prefinal_affine_dim, 1,
            stride=1, dilation=1, bias=True, padding=0
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
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
        
        # Just to get shape right for convolutions
        xs_pad = xs_pad.transpose(1, 2)
       
        # Basic pattern is (tdnn, relu, batchnorm, dropout) x num_layers
        # We also apply a skip connection using the bypass scale
        semi_ortho_step = self.training and (np.random.uniform(0,1) < 0.25)
        if semi_ortho_step:
            print("Taking step towards semi-orthogonality")
        for tdnnf, batchnorm in zip(self.tdnnf, self.batchnorm[:-1]):
            prev_xs_pad = xs_pad.detach().clone()
            xs_pad = self.nonlin(tdnnf(xs_pad, semi_ortho_step=semi_ortho_step))
            if not self.batch_norm_dropout: 
                xs_pad = batchnorm(xs_pad)
                xs_pad = F.dropout(xs_pad, p=self.dropout, training=self.training)
            if (xs_pad.shape == prev_xs_pad.shape):
                # only apply skip connection between TDNNF layers
                xs_pad = xs_pad + self.bypass_scale * prev_xs_pad
      
        if self.training:
            # print semi-orthogonal loss
            orth_error  = sum([x.orth_error() for x in self.tdnnf])
            print('Ortho: {}'.format(orth_error), end=' ')
        
        # A few final layers
        end_idx = xs_pad.size(2) if right_context == 0 else -right_context
        output2 = xs_pad.transpose(1, 2)[:, left_context:end_idx:self.subsample, :]
        xs_pad = self.nonlin(self.prefinal_affine(xs_pad))
        if not self.batch_norm_dropout:
            xs_pad = self.batchnorm[-1](xs_pad)
        
        # This is basically just glue
        output = xs_pad.transpose(1, 2)
        output = self.final_affine(output)
        return (
            output[:, left_context:end_idx:self.subsample, :],
            output2,
        )


class ChainTDNNF(TDNNF):
    @classmethod
    def build_model(cls, conf):
        model = ChainTDNNF(
            conf['idim'], conf['num_targets'],
            odims=[conf['tdnnf_hdim'] for i in range(conf['tdnnf_num_layers'])],
            bottleneck_dims=[conf['tdnnf_bottleneck_dim'] for i in range(conf['tdnnf_num_layers'])],
            dropout=conf['tdnnf_dropout'],
            bypass_scale=conf['tdnnf_bypass_scale'],
            prefinal_affine_dim=conf['tdnnf_prefinal_dim'],
            subsample=conf['subsample'],
        )   
        return model

    def __init__(
        self, idim, odim,
        odims=[625, 625, 625, 625, 625, 625],
        bottleneck_dims=[256, 256, 256, 256, 256, 256],
        prefinal_affine_dim=625,
        time_strides=[1,-1,1,-1,0,1,3,-3,3,-3,3,-3],
        nonlin=F.relu, dropout=0.1, bypass_scale=0.66, subsample=1,
    ):
        super().__init__(
            idim, odim, odims, bottleneck_dims, prefinal_affine_dim,
            time_strides, nonlin, dropout, bypass_scale, subsample
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
                xs_pad = self.xent_batchnorm(xs_pad.transpose(1, 2)).transpose(1, 2)
            xs_pad = self.xent_layer(xs_pad)
        return output, xs_pad 
