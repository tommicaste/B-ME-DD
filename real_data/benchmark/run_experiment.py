import os
import argparse
import datetime
import numpy as np
import torch
import pytorch_lightning as pl

from real_data.utils.config import DATASETS, MODELS
from real_data.utils.data import load_dataset, preprocess_data, split_data
from real_data.shared.targets import DEFAULT_TARGET
from real_data.utils.adjacency import apply_adjacency_strategy
from real_data.benchmark.training import train_shallow_models, train_deep_model, save_results


# ------- Argument Parsing -------

def parse_args():
    parser = argparse.ArgumentParser(description='Unified experiment runner')

    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--model', type=str, required=True, choices=MODELS.keys())
    parser.add_argument('--task', type=str, required=True, choices=['classification', 'regression'])

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time', type=str, default='max',
                        help="Window length T in samples, or 'max' for one window per scan.")
    parser.add_argument('--target', type=str, default=None,
                        help='Target column (adhd/abide) or window-fn (frankfurt). '
                             'Default per (task, dataset).')
    parser.add_argument('--n_mode', type=str, default='free', choices=['fixed', 'free'],
                        help="fixed=constant #windows/subject (anchored at cohort_source_T); free=all.")
    parser.add_argument('--stride_frac', type=float, default=1.0,
                        help='Stride as a fraction of T (1.0=non-overlap, 0.1=90%% overlap).')
    parser.add_argument('--tau', type=float, default=0.1)

    parser.add_argument('--parcellation', type=str, default=None,
                        help='Parcellation/atlas to use. If not set, uses the dataset default.')
    parser.add_argument('--roi_drop_frac', type=float, default=0.01,
                        help='abide only: drop ROIs that are zero-variance in at least this '
                             'fraction of scans, then drop the scans still carrying one. '
                             'Unset (default) keeps every ROI and loses those scans instead.')

    parser.add_argument('--conv_type', type=str, default=None,
                        choices=['gcn', 'sage', 'gin', 'gat'],
                        help='Convolution type for gnn model (default: run both gcn and sage)')
    parser.add_argument('--num_convs', type=int, default=None, choices=[1, 2],
                        help='When --conv_type is set, restrict to this many convolutions '
                             '(default: emits both 1-conv and 2-conv variants).')

    parser.add_argument('--adjacency', type=str, default='window',
                        choices=['window', 'bar', 'zero', 'circle', 'none'],
                        help='Adjacency strategy: window=per-window threshold, bar=per-subject '
                             'full-TS threshold, zero=mean of training-subject covariances, '
                             'circle=swap with non-overlapping window of same subject, '
                             'none=empty graph (no-convolution control; ignores --tau).')

    parser.add_argument('--shallow_models', type=str, nargs='+', default=None,
                        help='Subset of --model shallow estimators to fit. classification: '
                             'Ridge RandomForest SVM; regression: Ridge RandomForest SVR. '
                             'Default: all of them.')

    parser.add_argument('--activation', type=str, default='auto',
                        choices=['auto', 'relu', 'elu', 'identity'],
                        help="Post-convolution activation. auto = ELU for gat, ReLU otherwise. "
                             "identity makes a 1-conv SAGE/GCN model genuinely affine in the "
                             "features -- but NOT gat, whose attention softmax(LeakyReLU(.)) is "
                             "nonlinear regardless (measured: affinity residual 1.2 vs 4e-08 for "
                             "sage). Only the sage rows support a linear-model claim.")
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'vec'],
                        help="mean = global_mean_pool (standard). vec = keep every node "
                             "embedding; mean-pooling nearly cancels the adjacency.")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--es_monitor', type=str, default='auto',
                        choices=['auto', 'val_loss', 'val_acc'],
                        help="Metric that drives checkpoint selection and early stopping. "
                             "auto = val_acc for classification, val_loss for regression "
                             "(the historical behaviour). val_acc is discrete and noisy: in "
                             "our benchmark runs its argmax landed at epoch 0-1 in ~half of runs, so "
                             "the tested checkpoint was effectively untrained. Prefer val_loss.")
    parser.add_argument('--log_curves', action='store_true',
                        help='Write per-epoch val_loss/val_acc to a CSVLogger under '
                             '--out_dir. For diagnosing the stopping rule; off for grids.')
    parser.add_argument('--n_trials', type=int, default=50)

    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root of the raw datasets (default: $BMEDD_DATA_DIR, else ./data).')
    parser.add_argument('--out_dir', type=str, default=os.path.join('outputs', 'benchmark'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('outputs', 'checkpoints'))

    parser.add_argument('--fixed_params', type=str, default=None,
                        help='Path to JSON with frozen hyperparameters. If set, skips Optuna search.')
    parser.add_argument('--no_partner', action='store_true',
                        help='Drop the requirement that every window has a non-overlapping '
                             'partner (needed only by --adjacency circle). Keeps subjects whose '
                             'scans are too short for a partner at long T or small stride.')
    parser.add_argument('--cohort_source_T', type=int, default=None,
                        help='Reference window length for universal cohort filtering. When set, drops subjects '
                             'whose sessions are too short for circle adjacency at T=cohort_source_T. Applied '
                             'at load time so every run in a multi-T sweep sees the same cohort.')

    return parser.parse_args()


# ------- Main -------

def main():
    args = parse_args()
    if args.data_dir is not None:
        os.environ['BMEDD_DATA_DIR'] = args.data_dir

    args.run_timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    args._collected_thresholds = []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.target is None:
        args.target = DEFAULT_TARGET[args.task][args.dataset]

    print(f"Running experiment: {args.dataset} / {args.model} / {args.task} / target={args.target}")
    print(f"Seed: {args.seed}, Time: {args.time}, n_mode: {args.n_mode}, stride_frac: {args.stride_frac}")

    if args.cohort_source_T is not None and args.time != 'max' \
            and int(args.cohort_source_T) < int(args.time):
        raise ValueError(
            f"cohort_source_T={args.cohort_source_T} < time={args.time}. "
            f"The fixed-n anchor must be >= the window length."
        )

    if args.es_monitor == 'val_acc' and args.task == 'regression':
        raise ValueError("--es_monitor val_acc needs --task classification; "
                         "val_acc is never logged for regression.")

    model_config = MODELS[args.model]

    if args.model == 'gnn' and args.conv_type is not None:
        n_list = [args.num_convs] if args.num_convs is not None else [1, 2]
        model_config = {**model_config, 'variants': [(args.conv_type, n) for n in n_list]}

    raw_data = load_dataset(args)
    print(f"Loaded {len(raw_data)} subjects")

    data = preprocess_data(raw_data, args, model_config)
    splits = split_data(data, args)

    if model_config['needs_adjacency']:
        splits = apply_adjacency_strategy(data, splits, args)

    if model_config['is_deep']:
        results = train_deep_model(splits, args, model_config)
    else:
        results = train_shallow_models(splits, args)

    if args._collected_thresholds:
        args._effective_threshold_mean = float(np.mean(args._collected_thresholds))
    else:
        args._effective_threshold_mean = None

    save_results(results, splits, args)

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
