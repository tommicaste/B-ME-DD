import os
import glob
import json
import shutil
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier, DummyRegressor
import optuna

from real_data.benchmark.models import LR, MLP, SparseGNNRegressor, LightningModel, GraphDataset, PyGGraphDataset, LightningModelPyG, BrainGNNWrapper, LightningBrainGNN
from real_data.benchmark.neurograph import ResidualGNNs
from real_data.shared.evaluation import score_both, git_provenance
from torch_geometric.loader import DataLoader as PyGDataLoader


# ------- Run Tag -------

# Path-only args: they change where output lands, never what it contains, so two runs
# differing only in these are the SAME experiment and must hash identically.
# `fixed_params` is deliberately NOT here: it is a path, but it replaces the entire
# hyperparameter search, so two runs differing only in it are different experiments.
_PATH_ONLY = {'out_dir', 'checkpoint_dir'}


def config_dict(args):
    """Every argument that can change a result, canonicalised.

    `fixed_params` is folded in by CONTENT, not by path -- otherwise a run frozen
    at weight_decay=0 and one frozen at 1e-3 hash identically, which would let
    them share a checkpoint dir and make their rows indistinguishable afterwards.
    """
    d = {k: v for k, v in sorted(vars(args).items())
         if not k.startswith('_') and k not in _PATH_ONLY
         and k != 'run_timestamp' and not callable(v)}
    fp = d.get('fixed_params')
    if fp:
        try:
            d['fixed_params'] = json.loads(Path(fp).read_text())
        except (OSError, ValueError):
            pass
    return d


def config_hash(args):
    """Short digest of the full config.

    Written on every row and appended to the checkpoint tag. This is what makes two
    runs differing in ANY argument distinguishable after the fact and stops their
    checkpoint dirs colliding -- including arguments nobody remembered to log.
    Pairs with git_provenance(): config_hash pins the ARGUMENTS, git_sha the CODE,
    and a row needs both to be reproducible.
    """
    payload = json.dumps(config_dict(args), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def resolve_monitor(args):
    """(metric, mode) driving ModelCheckpoint and EarlyStopping.

    'auto' reproduces the historical behaviour: val_acc for classification,
    val_loss for regression. val_acc is a poor monitor -- it is discrete and
    high-variance, so its argmax lands in the first epoch or two by noise and the
    checkpoint that gets tested is effectively untrained (measured in our
    benchmark runs: median best epoch 1-5, no run past epoch 65 of a possible
    200). val_loss is continuous and does not have this failure mode.
    """
    m = getattr(args, 'es_monitor', 'auto')
    if m == 'auto':
        m = 'val_acc' if args.task == 'classification' else 'val_loss'
    return m, ('max' if m == 'val_acc' else 'min')


_ACT = {'relu': nn.ReLU, 'elu': nn.ELU, 'identity': nn.Identity}


def resolve_activation(args, kind):
    """Return (module, name). `auto` follows the per-conv default so old rows, which
    predate the flag, are interpretable as exactly what `auto` resolves to."""
    a = getattr(args, 'activation', 'auto')
    if a == 'auto':
        a = 'elu' if kind.lower() == 'gat' else 'relu'
    return _ACT[a](), a


def make_logger(args, variant_name):
    """CSVLogger when --log_curves is set, otherwise no logger.

    The 1,640-job grid does not want per-epoch traces, but without them there is
    no record anywhere of how val_loss and val_acc actually evolve -- and that is
    exactly what a stopping-rule question needs. Run a handful of jobs with
    --log_curves --patience 200 to see the curves before trusting a stopping rule.
    """
    if not getattr(args, 'log_curves', False):
        return False
    return pl.loggers.CSVLogger(save_dir=args.out_dir, name=f"curves_{variant_name}")


def _run_tag(args, variant_name=None):
    """Build a deterministic, readable string encoding every arg that affects
    the model weights or data content. Used in checkpoint dirpaths and output
    filenames so concurrent runs with different settings never collide.

    Excludes path-only args (out_dir, checkpoint_dir). When variant_name is
    provided, it is appended (used for checkpoint paths). When omitted, the tag
    is shared across variants of the same experiment (used for the output
    filename where variants are separate rows).
    """
    parcellation = getattr(args, 'parcellation', None) or "default"
    adjacency = getattr(args, 'adjacency', 'window')
    parts = [
        args.dataset,
        args.task,
        f"adj-{adjacency}",
        f"seed{args.seed}",
        f"tau{args.tau:.4f}",
        f"time{args.time}",
        f"nmode-{args.n_mode}",
        f"sfrac{args.stride_frac}",
        f"par-{parcellation}",
        f"bs{args.batch_size}",
        f"ep{args.max_epochs}",
        f"pat{args.patience}",
        f"trials{args.n_trials}",
        config_hash(args),
    ]
    if variant_name is not None:
        parts.insert(2, variant_name)
    return "_".join(str(p) for p in parts)


def _output_filename(args):
    return f"{args.dataset}_{args.task}_seed{args.seed}.jsonl"


# ------- Optuna Pruning -------

class OptunaPruningCallback(pl.Callback):
    """Report the monitored metric to Optuna each epoch and prune losing trials.

    `sign` maps the metric into the study's direction (always "maximize"), so it
    must match what `objective` returns: +1 for val_acc, -1 for val_loss.
    """

    def __init__(self, trial, monitor, sign):
        self.trial, self.monitor, self.sign = trial, monitor, sign

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(self.sign * float(value), trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"trial {self.trial.number} pruned at epoch {trainer.current_epoch}")


# n_startup_trials matches TPESampler's 10 random startup trials, so nothing is
# pruned against an empty baseline. n_warmup_steps must sit BELOW the early-stopping
# horizon or the pruner never acts: at patience=15 a stagnating trial is killed by
# EarlyStopping around epoch 16-30, so a warmup of 20 would fire only after the fact.
# 10 epochs still gives a slow-starting config (low lr, high weight decay) room to
# show signal before it is judged.
def make_pruner():
    return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10,
                                       interval_steps=5)


# ------- Shallow Model Training -------

def _select_shallow(models_config, args):
    """Restrict --model shallow to the estimators named by --shallow_models."""
    sel = getattr(args, 'shallow_models', None)
    if not sel:
        return models_config
    unknown = sorted(set(sel) - set(models_config))
    if unknown:
        raise ValueError(f"--shallow_models {unknown} unavailable for task={args.task}; "
                         f"choose from {sorted(models_config)}")
    return {k: v for k, v in models_config.items() if k in sel}


def _decision_scores(model, X):
    """(continuous score, its threshold) for a fitted sklearn classifier.

    decision_function is preferred because it is what .predict() follows for
    RidgeClassifier (splits at 0) and for SVC (which ignores its own Platt-scaled
    predict_proba), so the window-level accuracy from these scores reproduces
    .predict() exactly. RandomForest has no decision_function and falls back to
    the class-1 probability at 0.5, which reproduces .predict() everywhere except
    an exact 0.5 tie, where argmax takes class 0 and `>= 0.5` takes class 1.
    """
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1), 0.0
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float), 0.5


def train_shallow_models(splits, args):
    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_val, y_val = splits['val']['X'], splits['val']['y']
    X_test, y_test = splits['test']['X'], splits['test']['y']

    # Refit on train only, NOT train+val. The deep models fit on train and spend val
    # on early stopping / checkpoint selection (training.py, final_trainer.fit), so
    # refitting the shallow baselines on train+val would hand them ~33% more windows
    # than every GNN they are meant to be compared against -- a bias pointing at the
    # very conclusion this benchmark tests.

    results = {}

    if args.task == 'classification':
        models_config = {
            'Ridge': {
                'param_grid': [{'alpha': a} for a in np.logspace(-2, 6, 30)],
                'model_fn': lambda p: make_pipeline(StandardScaler(), RidgeClassifier(**p, max_iter=1000)),
            },
            'RandomForest': {
                'param_grid': [
                    {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml, "max_features": "sqrt"}
                    for ne in (200, 400, 800)
                    for md in (None, 10, 20)
                    for ml in (1, 5)
                ],
                'model_fn': lambda p: RandomForestClassifier(random_state=args.seed, n_jobs=-1, **p),
            },
            'SVM': {
                'param_grid': [
                    {"C": c, "gamma": g}
                    for c in (0.1, 1, 10, 100)
                    for g in ("scale", "auto")
                ],
                'model_fn': lambda p: make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, **p)),
            },
        }

        models_config = _select_shallow(models_config, args)

        # Select on accuracy, the metric that is reported and that the deep models
        # monitor (val_acc) -- selecting on AUC here biased the graph-agnostic
        # baselines against the GNNs they exist to be compared with. predict() is
        # used rather than thresholding a score: RidgeClassifier's decision
        # function splits at 0, not 0.5, and SVC.predict follows the SVM decision
        # function rather than its Platt-scaled predict_proba.
        for name, cfg in models_config.items():
            print(f"Training {name}...")
            best_val_acc, best_params = -np.inf, None

            for params in cfg['param_grid']:
                model = cfg['model_fn'](params)
                model.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, model.predict(X_val))
                if val_acc > best_val_acc:
                    best_val_acc, best_params = val_acc, params

            final_model = cfg['model_fn'](best_params)
            final_model.fit(X_train, y_train)

            test_score, test_thresh = _decision_scores(final_model, X_test)
            results[name] = {
                'best_val': best_val_acc,
                'test_acc': accuracy_score(y_test, final_model.predict(X_test)),
                'best_params': best_params,
                '_test_score': test_score,
                '_test_threshold': test_thresh,
            }
            print(f"  Best val acc: {best_val_acc:.4f}, Test acc: {results[name]['test_acc']:.4f}")

    else:
        models_config = {
            'Ridge': {
                'param_grid': [{'alpha': a} for a in np.logspace(-2, 6, 30)],
                'model_fn': lambda p: make_pipeline(StandardScaler(), Ridge(**p)),
            },
            'RandomForest': {
                'param_grid': [
                    {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml}
                    for ne in (200, 400, 800)
                    for md in (None, 10, 20)
                    for ml in (1, 5)
                ],
                'model_fn': lambda p: RandomForestRegressor(random_state=args.seed, n_jobs=-1, **p),
            },
            'SVR': {
                'param_grid': [{"C": c, "gamma": g} for c in (0.1, 1, 10, 100) for g in ("scale", "auto")],
                'model_fn': lambda p: make_pipeline(StandardScaler(), SVR(kernel="rbf", **p)),
            },
        }

        models_config = _select_shallow(models_config, args)

        for name, cfg in models_config.items():
            print(f"Training {name}...")
            best_val_mse, best_params = np.inf, None

            for params in cfg['param_grid']:
                model = cfg['model_fn'](params)
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, val_pred)
                if val_mse < best_val_mse:
                    best_val_mse, best_params = val_mse, params

            final_model = cfg['model_fn'](best_params)
            final_model.fit(X_train, y_train)

            test_pred = final_model.predict(X_test)
            results[name] = {
                'best_val': best_val_mse,
                'test_mse': mean_squared_error(y_test, test_pred),
                'best_params': best_params,
                '_test_score': np.asarray(test_pred, dtype=float),
                '_test_threshold': None,
            }
            print(f"  Best val MSE: {best_val_mse:.4f}, Test MSE: {results[name]['test_mse']:.4f}")

    return results


# ------- Deep Model Training -------

def train_deep_model(splits, args, model_config):
    use_cuda = torch.cuda.is_available()
    accelerator = "gpu" if use_cuda else "cpu"
    print(f"Using: {accelerator}")

    use_pyg = args.model in ('gnn', 'residualgnn', 'braingnn')

    if use_pyg:
        # PyG path: Create Data objects with edge_index
        include_edge_attr = (args.model == 'braingnn')

        def make_pyg_loader(split_data, shuffle=False):
            dataset = PyGGraphDataset(
                x=split_data['C'],
                y=split_data['y'],
                adjacency=split_data['A'],
                include_edge_attr=include_edge_attr,
            )
            # drop_last on train only: n_train % batch_size == 1 makes BatchNorm1d raise
            # ("Expected more than 1 value per channel"). val/test keep every sample.
            return PyGDataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                 num_workers=0, drop_last=shuffle)

        train_loader = make_pyg_loader(splits['train'], shuffle=True)
        val_loader = make_pyg_loader(splits['val'])
        test_loader = make_pyg_loader(splits['test'])
        input_dim = splits['train']['C'].shape[2]  # n_rois

    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits['train']['X'])
        X_val = scaler.transform(splits['val']['X'])
        X_test = scaler.transform(splits['test']['X'])

        train_data = (X_train, None, splits['train']['y'])
        val_data = (X_val, None, splits['val']['y'])
        test_data = (X_test, None, splits['test']['y'])
        input_dim = X_train.shape[1]

        def make_loader(data, shuffle=False):
            X, A, y = data
            dataset = GraphDataset(X, y, adjacency=A)
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                              num_workers=0, pin_memory=use_cuda, drop_last=shuffle)

        train_loader = make_loader(train_data, shuffle=True)
        val_loader = make_loader(val_data)
        test_loader = make_loader(test_data)

    variants = model_config['variants'] or [(None, None)]

    fixed_params_path = getattr(args, 'fixed_params', None)
    fixed_params = None
    if fixed_params_path:
        fixed_params = json.loads(Path(fixed_params_path).read_text())
        print(f"Using fixed params from {fixed_params_path}: {fixed_params}")

    all_results = {}

    for variant in variants:
        variant_name = f"{variant[0]}_{variant[1]}conv" if variant[0] else args.model
        print(f"\nTraining {variant_name}...")

        def objective(trial):
            pl.seed_everything(args.seed, workers=True)


            # AdamW's decoupled decay shrinks weights by exp(-lr*wd*N_steps), so the
            # scale that matters is lambda_eff = lr*wd*N. At lr=2e-3 the old ceiling
            # of wd=1e-3 gives 0.03% shrinkage at the epoch these runs actually stop
            # and only 1.2% at the full 200 epochs -- it could not regularize anything
            # at any budget this pipeline can reach. Reaching lambda_eff ~ 1 needs
            # wd ~ 1e-1, so the ceiling is raised three orders of magnitude.
            lr = trial.suggest_float("lr", 3e-4, 1e-2, log=True)
            wd = trial.suggest_float("weight_decay", 1e-4, 3e-1, log=True)
            l1 = trial.suggest_categorical("l1_lambda", [0.0, 1e-5, 1e-4, 1e-3])


            if args.model in ('gnn', 'mlp'):
                # Capacity is not the binding constraint: with weight_decay searched up
                # to 3e-1 the pre-pass picked 128 and 256 and reached 0.67 subject
                # accuracy, where the same widths at wd=0 memorised in one epoch.
                hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
            elif args.model == 'lr':
                hidden_dim = 64
            elif args.model in ('residualgnn', 'braingnn'):
                hidden_dim = None  # Not used for these models
            else:
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

            if args.model == 'gnn':
                act, _ = resolve_activation(args, variant[0])
                model = SparseGNNRegressor(variant[0], input_dim, hidden_dim, variant[1],
                                           activation=act,
                                           readout=getattr(args, 'readout', 'mean'))
            elif args.model == 'mlp':
                dropout = trial.suggest_categorical("dropout", [0.0, 0.5])
                model = MLP(input_dim, hidden_dim, dropout=dropout)
            elif args.model == 'residualgnn':
                hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128])
                hidden = trial.suggest_categorical("hidden", [128, 256])
                num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
                gnn_type = "SAGEConv"  # architecture choice, not swept (paper's ResidualGNN)

                model = ResidualGNNs(
                    num_features=input_dim,
                    hidden_channels=hidden_channels,
                    hidden=hidden,
                    num_layers=num_layers,
                    num_classes=1,
                    gnn_type=gnn_type,
                )
            elif args.model == 'braingnn':
                ratio = trial.suggest_categorical("ratio", [0.3, 0.5])
                k = trial.suggest_categorical("k", [8, 16])
                lamb3 = trial.suggest_categorical("lamb3", [0.1, 0.3])
                lamb5 = trial.suggest_categorical("lamb5", [0.1, 0.3])

                model = BrainGNNWrapper(
                    indim=input_dim,
                    ratio=ratio,
                    k=k,
                    R=input_dim,
                )
            else:
                model = LR(input_dim)

            if args.model in ('gnn', 'residualgnn'):
                lit_model = LightningModelPyG(
                    model=model,
                    task=args.task,
                    lr=lr,
                    weight_decay=wd,
                    l1_lambda=l1,
                )
            elif args.model == 'braingnn':
                lit_model = LightningBrainGNN(
                    model=model,
                    task=args.task,
                    lr=lr,
                    weight_decay=wd,
                    ratio=ratio,
                    lamb3=lamb3,
                    lamb5=lamb5,
                )
            else:
                lit_model = LightningModel(
                    model=model,
                    task=args.task,
                    lr=lr,
                    weight_decay=wd,
                    l1_lambda=l1,
                    use_adjacency=model_config['needs_adjacency'],
                )

            monitor, mode = resolve_monitor(args)

            checkpoint_cb = pl.callbacks.ModelCheckpoint(
                monitor=monitor, mode=mode, save_top_k=1,
                dirpath=f"{args.checkpoint_dir}/temp_{_run_tag(args, variant_name)}_trial{trial.number}_pid{os.getpid()}",
            )

            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=1,
                max_epochs=args.max_epochs,
                callbacks=[
                    pl.callbacks.EarlyStopping(monitor=monitor, patience=args.patience, mode=mode),
                    checkpoint_cb,
                    OptunaPruningCallback(trial, monitor, 1.0 if mode == 'max' else -1.0),
                ],
                enable_checkpointing=True,
                enable_progress_bar=False,
                logger=False,
            )

            trainer.fit(lit_model, train_loader, val_loader)

            score = checkpoint_cb.best_model_score.item()
            return score if mode == 'max' else -score      # the study always maximizes

        if fixed_params is not None:
            best_params = dict(fixed_params)
            study = None
            print(f"Skipping Optuna. Using fixed params: {best_params}")
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                pruner=make_pruner(),
            )
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params
            print(f"Best params: {best_params}")

        pl.seed_everything(args.seed, workers=True)

        if args.model == 'gnn':
            act, _ = resolve_activation(args, variant[0])
            final_model = SparseGNNRegressor(variant[0], input_dim, best_params["hidden_dim"],
                                             variant[1], activation=act,
                                             readout=getattr(args, 'readout', 'mean'))
        elif args.model == 'mlp':
            final_model = MLP(input_dim, best_params["hidden_dim"], dropout=best_params["dropout"])
        elif args.model == 'residualgnn':
            final_model = ResidualGNNs(
                num_features=input_dim,
                hidden_channels=best_params["hidden_channels"],
                hidden=best_params["hidden"],
                num_layers=best_params["num_layers"],
                num_classes=1,
                gnn_type="SAGEConv",
            )
        elif args.model == 'braingnn':
            final_model = BrainGNNWrapper(
                indim=input_dim,
                ratio=best_params["ratio"],
                k=best_params["k"],
                R=input_dim,
            )
        else:
            final_model = LR(input_dim)

        if args.model in ('gnn', 'residualgnn'):
            final_lit_model = LightningModelPyG(
                model=final_model,
                task=args.task,
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
                l1_lambda=best_params["l1_lambda"],
            )
        elif args.model == 'braingnn':
            final_lit_model = LightningBrainGNN(
                model=final_model,
                task=args.task,
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
                ratio=best_params["ratio"],
                lamb3=best_params["lamb3"],
                lamb5=best_params["lamb5"],
            )
        else:
            final_lit_model = LightningModel(
                model=final_model,
                task=args.task,
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
                l1_lambda=best_params["l1_lambda"],
                use_adjacency=model_config['needs_adjacency'],
            )

        monitor, mode = resolve_monitor(args)

        final_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode=mode, save_top_k=1,
            dirpath=f"{args.checkpoint_dir}/{_run_tag(args, variant_name)}_pid{os.getpid()}",
        )

        final_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=args.max_epochs,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor=monitor, patience=args.patience, mode=mode),
                final_checkpoint
            ],
            enable_checkpointing=True,
            enable_progress_bar=False,
            logger=make_logger(args, variant_name),
        )

        final_trainer.fit(final_lit_model, train_loader, val_loader)
        test_results = final_trainer.test(final_lit_model, test_loader, ckpt_path="best")

        # Raw per-sample test scores, in loader order (test_loader is unshuffled and
        # keeps every sample), so save_results can score per subject as well as per
        # window. Must run BEFORE the checkpoint cleanup below, which deletes "best".
        test_score = torch.cat([
            p.detach().float().cpu().reshape(-1)
            for p in final_trainer.predict(final_lit_model, test_loader, ckpt_path="best")
        ]).numpy()

        # Cleanup: remove all temp trial checkpoints AND the final checkpoint
        # for this variant. Test metrics are already captured in test_results,
        # so nothing downstream needs the on-disk checkpoints.
        _run_tag_variant = f"{_run_tag(args, variant_name)}_pid{os.getpid()}"
        temp_dirs = glob.glob(f"{args.checkpoint_dir}/temp_{_run_tag(args, variant_name)}_trial*_pid{os.getpid()}")
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(f"{args.checkpoint_dir}/{_run_tag_variant}", ignore_errors=True)
        print(f"Cleaned {len(temp_dirs)} temp + 1 final checkpoint dir for {variant_name}")

        if study is not None:
            best_val_for_row = study.best_value if mode == 'max' else -study.best_value
        else:
            score = final_checkpoint.best_model_score
            best_val_for_row = float(score.item()) if score is not None else float('nan')

        if args.task == 'classification':
            all_results[variant_name] = {
                'best_val': best_val_for_row,
                'test_acc': test_results[0]['test_acc'],
                'best_params': best_params,
                '_test_score': test_score,     # logits: BCEWithLogitsLoss, so split at 0
                '_test_threshold': 0.0,
            }
            print(f"Test accuracy: {test_results[0]['test_acc']:.4f}")
        else:
            all_results[variant_name] = {
                'best_val': best_val_for_row,
                'test_mse': test_results[0]['test_loss'],
                'best_params': best_params,
                '_test_score': test_score,
                '_test_threshold': None,
            }
            print(f"Test MSE: {test_results[0]['test_loss']:.4f}")

    return all_results


# ------- Results Saving -------

def save_results(results, splits, args):
    os.makedirs(args.out_dir, exist_ok=True)
    adjacency = getattr(args, 'adjacency', 'window')
    output_file = os.path.join(args.out_dir, _output_filename(args))

    eval_level = getattr(args, 'eval_level', 'window')

    base_info = {
        'timestamp': getattr(args, 'run_timestamp', None),
        **git_provenance(),
        'dataset': args.dataset,
        'model': args.model,
        'task': args.task,
        'seed': args.seed,
        'time': args.time,
        'target': args.target,
        'eval_level': eval_level,
        'n_mode': args.n_mode,
        'stride_frac': args.stride_frac,
        'tau': args.tau,
        'adjacency': adjacency,
        'cohort_source_T': getattr(args, 'cohort_source_T', None),
        'effective_threshold_mean': getattr(args, '_effective_threshold_mean', None),
        'conv_type': getattr(args, 'conv_type', None),
        'num_convs': getattr(args, 'num_convs', None),
        'n_train': len(splits['train']['y']),
        'n_val': len(splits['val']['y']),
        'n_test': len(splits['test']['y']),
        'n_subjects_train': len(np.unique(splits['train']['groups'])),
        'n_subjects_val': len(np.unique(splits['val']['groups'])),
        'n_subjects_test': len(np.unique(splits['test']['groups'])),
    }

    base_info['parcellation'] = args.parcellation
    # best_val is whatever the monitor optimised, so its UNITS depend on es_monitor:
    # accuracy under val_acc, loss under val_loss. Record both so the column is
    # never ambiguous after the fact.
    base_info['es_monitor'], base_info['es_mode'] = resolve_monitor(args)
    kind = getattr(args, 'conv_type', None) or ''
    base_info['activation'] = resolve_activation(args, kind)[1]
    base_info['readout'] = getattr(args, 'readout', 'mean')
    base_info['config_hash'] = config_hash(args)
    base_info['config'] = config_dict(args)

    # Null reference for this split, via sklearn's dummy estimators. Recorded per
    # row rather than run as a model: it is
    # a property of the split, so it costs nothing and cannot desync from it.
    y_tr, y_te = splits['train']['y'], splits['test']['y']
    g_te = splits['test']['groups']
    z_tr, z_te = np.zeros((len(y_tr), 1)), np.zeros((len(y_te), 1))
    if args.task == 'classification':
        dummy = DummyClassifier(strategy='most_frequent').fit(z_tr, y_tr)
        null_pred = dummy.predict(z_te).astype(float)
        base_info['null_acc'] = float(accuracy_score(y_te, null_pred))
    else:
        dummy = DummyRegressor(strategy='mean').fit(z_tr, y_tr)
        null_pred = dummy.predict(z_te).astype(float)
        base_info['null_mse'] = float(mean_squared_error(y_te, null_pred))
    for k, v in score_both(y_te, null_pred, g_te, args.task, eval_level).items():
        base_info[f'null_{k}'] = v

    def expand(res):
        """Replace the raw test scores with metrics at both eval levels.

        `test_acc` / `test_mse` are left exactly as the training loop produced
        them, so rows stay comparable with earlier runs; the *_win / *_subj keys
        are the normalized pair, and the unsuffixed acc/auc/R2/MSE hold whichever
        level this target declares primary.
        """
        res = dict(res)
        s = res.pop('_test_score', None)
        thr = res.pop('_test_threshold', None)
        if s is None or len(s) != len(y_te):
            return res
        kw = {} if thr is None else {'threshold': thr}
        return {**res, **score_both(y_te, s, g_te, args.task, eval_level, **kw)}

    with open(output_file, "a") as f:
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            for variant_name, variant_results in results.items():
                record = {**base_info, 'variant': variant_name, **expand(variant_results)}
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        else:
            record = {**base_info, **expand(results)}
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"Results saved to: {output_file}")
