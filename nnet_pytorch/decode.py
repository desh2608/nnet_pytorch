#!/usr/bin/env python

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import datasets
import models
from LRScheduler import LRScheduler
from batch_generators import evaluation_batches
from IterationTypes import decode_dataset
import kaldi_io


def main():
    args = parse_arguments()
    print(args)

    # Reserve the GPU if used in decoding. In general it won't be.        
    if args.gpu:
        # User will need to set CUDA_VISIBLE_DEVICES here
        cvd = subprocess.check_output(["/usr/bin/free-gpu", "-n", "1"]).decode().strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    reserve_variable = torch.ones(1).to(device)

    # Load experiment configurations so that decoding uses the same parameters
    # as training
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    dataset_args = eval(conf['datasets'])[0]
    
    # Load the decoding dataset
    subsample_val = 1
    if 'subsample' in conf:
        subsample_val=conf['subsample']
   
    # Note here that the targets file is just a dummy placeholder. We don't
    # need the targets or use them. The keyword argument validation=0, is
    # because we are decoding and do not have the targets so validation does
    # not make sense in this context.
    targets = os.path.join(args.datadir, 'pdfid.{}.tgt'.format(str(subsample_val)))
    if not os.path.exists(targets):
        print("Dummy targets not found")
        sys.exit(1)
    
    # Update the dataset dictionary to reflect evaluation set
    dataset_args.update(
        {'data':args.datadir, 'tgt':targets, 'subsample': subsample_val,
        'ivector_dim': args.ivector_dim, 'utt_subset': args.utt_subset})
    if args.ivector_scp is not None:
        dataset_args.update({'ivectors':args.ivector_scp})
    if args.aux_targets is not None:
        dataset_args.update({'aux_tgt': args.aux_targets})
    conf.update({'datasets':[dataset_args]})
    dataset = datasets.DATASETS[conf['datasetname']].build_dataset(dataset_args)

    # We just need to add in the input dimensions. This depends on the type of
    # features used.
    conf['idim'] = args.idim
    
    print(conf) 
    # Build the model and send to the device (cpu or gpu). Generally cpu.
    model = models.MODELS[conf['model']].build_model(conf)
    model.to(device)
  
    # Load the model from experiment checkpoint 
    mdl = torch.load(
        os.path.sep.join([args.modeldir, args.checkpoint]),
        map_location=device
    )
    model.load_state_dict(mdl['model'])
   
    # Load the state priors (For x-ent only systems)
    priors = 0.0
    if 'LFMMI' not in conf['objective']:
        priors = json.load(open(os.path.join(args.modeldir, args.prior_name)))
        priors = np.array(priors)
    
        # Floor likelihoods (by altering the prior) for states with very low
        # priors
        priors[priors < args.prior_floor] = 1e20
    args.objective = conf['objective']
    args.datasetname = conf['datasetname']
    decode(args, dataset, model, priors, device=device)


def decode(args, dataset, model, priors, device='cpu'):
    '''
        Produce lattices from the input utterances.
    '''
    # This is all of the kaldi code we are calling. We are just piping out
    # out features to latgen-faster-mapped which does all of the lattice
    # generation.
    lat_output = '''ark:| copy-feats ark:- ark:- |\
    latgen-faster-mapped --min-active={} --max-active={} \
    --max-mem={} --prune-interval={} \
    --lattice-beam={} --beam={} \
    --acoustic-scale={} --allow-partial=true \
    --word-symbol-table={} \
    {} {} ark:- ark:- | lattice-scale --acoustic-scale={} ark:- ark:- |\
    gzip -c > {}/lat.{}.gz'''.format(
        args.min_active, args.max_active, args.max_mem, args.prune_interval,
        args.lattice_beam, args.beam, args.acoustic_scale,
        args.words_file, args.trans_mdl, args.hclg,
        args.post_decode_acwt, args.dumpdir, args.job
    )
    post_output = {}
    entropy = {}
    max_prob = {}

    # Do the decoding (dumping senone posteriors)
    model.eval()
    with torch.no_grad():
        with kaldi_io.open_or_fd(lat_output, 'wb') as f:
            utt_mat = [] 
            prev_key = b''
            generator = evaluation_batches(dataset)
            # Each minibatch is guaranteed to have at most 1 utterance. We need
            # to append the output of subsequent minibatches corresponding to
            # the same utterances. These are stored in ``utt_mat'', which is
            # just a buffer to accumulate the posterior outputs of minibatches
            # corresponding to the same utterance. The posterior state
            # probabilities are normalized (subtraction in log space), by the
            # log priors in order to produce pseudo-likelihoods useable for
            # for lattice generation with latgen-faster-mapped
            for key, mat in decode_dataset(args, generator, model, device='cpu'):
                if len(utt_mat) > 0 and key != prev_key:   
                    kaldi_io.write_mat(
                        f, np.concatenate(utt_mat, axis=0)[:utt_length, :],
                        key=prev_key.decode('utf-8')
                    )
                    post_output[prev_key.decode('utf-8')] = np.concatenate(utt_mat, axis=0)[:utt_length, :].tolist()
                    entropy[prev_key.decode('utf-8')] = compute_per_frame_entropy(np.concatenate(utt_mat, axis=0)[:utt_length, :]).tolist()
                    max_prob[prev_key.decode('utf-8')] = compute_per_frame_max(np.concatenate(utt_mat, axis=0)[:utt_length, :]).tolist()
                    utt_mat = []
                utt_mat.append(mat - args.prior_scale * priors)
                prev_key = key
                utt_length = dataset.utt_lengths[key] // dataset.subsample 

            # Flush utt_mat buffer at the end
            if len(utt_mat) > 0:
                kaldi_io.write_mat(
                    f,
                    np.concatenate(utt_mat, axis=0)[:utt_length, :],
                    key=prev_key.decode('utf-8')
                )
                post_output[prev_key.decode('utf-8')] = np.concatenate(utt_mat, axis=0)[:utt_length, :].tolist()
                entropy[prev_key.decode('utf-8')] = compute_per_frame_entropy(np.concatenate(utt_mat, axis=0)[:utt_length, :]).tolist()
                max_prob[prev_key.decode('utf-8')] = compute_per_frame_max(np.concatenate(utt_mat, axis=0)[:utt_length, :]).tolist()

    if args.save_post:
        with open('{}/post.{}.json'.format(args.dumpdir,args.job),'w') as fp, \
            open('{}/entropy.{}.json'.format(args.dumpdir,args.job), 'w') as fe, \
            open('{}/max_prob.{}.json'.format(args.dumpdir,args.job), 'w') as fm:
            json.dump(post_output, fp)
            json.dump(entropy, fe)
            json.dump(max_prob, fm)

def compute_per_frame_entropy(posteriors):
    # Transform posteriors into probability distribution
    probs = np.exp(posteriors)/np.exp(posteriors).sum(axis=1, keepdims=True)
    H = np.sum(-probs * np.log2(probs), axis=1)
    return H

def compute_per_frame_max(posteriors):
    probs = np.exp(posteriors)/np.exp(posteriors).sum(axis=1, keepdims=True)
    return np.max(probs, axis=1)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--modeldir')
    parser.add_argument('--dumpdir')
    parser.add_argument('--checkpoint', default='final.mdl')
    parser.add_argument('--idim', type=int)
    parser.add_argument('--ivector-dim', type=int, default=None)
    parser.add_argument('--ivector-scp', type=str, default=None)
    parser.add_argument('--aux-targets', type=str, default=None)
    parser.add_argument('--prior-scale', type=float, default=1.0)
    parser.add_argument('--prior-floor', type=float, default=-20)
    parser.add_argument('--prior-name', default='priors')
    parser.add_argument('--words-file')
    parser.add_argument('--trans-mdl')
    parser.add_argument('--hclg')
    parser.add_argument('--min-active', type=int, default=200)
    parser.add_argument('--max-active', type=int, default=7000)
    parser.add_argument('--max-mem', type=int, default=50000000)
    parser.add_argument('--lattice-beam', type=float, default=8.0)
    parser.add_argument('--beam', type=float, default=15.0)
    parser.add_argument('--prune-interval', type=int, default=25)
    parser.add_argument('--acoustic-scale', type=float, default=0.1)
    parser.add_argument('--post-decode-acwt', type=float, default=1.0)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--save-post', action='store_true')
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    datasets.DATASETS[conf['datasetname']].add_args(parser)
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args


if __name__ == "__main__":
    main() 
