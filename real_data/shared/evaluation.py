"""Scoring + run provenance shared by every real-data experiment.

Every run scores the SAME predictions at BOTH levels:

  window   one row per window (a subject with K windows counts K times)
  subject  predictions averaged within subject first, then scored

`resolve_target` declares which level is primary for a given target
(`subject` for the static adhd/abide phenotypes, `window` for frankfurt's
per-window sleep labels). `score_both` emits the primary under the plain metric
name and both levels under `_win` / `_subj`, so a reader never has to infer
which quantity a column holds.
"""
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score


def per_subject(y, pred, groups):
    uniq = np.unique(groups)
    true_s = np.array([y[groups == s].mean() for s in uniq])
    pred_s = np.array([pred[groups == s].mean() for s in uniq])
    return true_s, pred_s


def score(y, pred, groups, eval_level, task, threshold=0.5):
    """Per-subject (average scores within subject) or per-window scoring.

    `pred` is whatever continuous score the model emits and `threshold` is that
    model's own decision boundary -- 0.5 for a class-1 probability, 0.0 for a
    logit or an sklearn decision_function. Aggregating the score and then
    applying the threshold (rather than voting over hard labels) is what makes
    the subject-level number comparable across estimator families.
    """
    if eval_level == 'subject':
        yt, yp = per_subject(np.asarray(y), np.asarray(pred), np.asarray(groups))
    else:
        yt, yp = np.asarray(y), np.asarray(pred)
    if task == 'regression':
        return {'R2': float(r2_score(yt, yp)), 'MSE': float(mean_squared_error(yt, yp))}
    yt = np.rint(yt).astype(int)
    acc = float(accuracy_score(yt, (yp >= threshold).astype(int)))
    auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float('nan')
    return {'acc': acc, 'auc': auc}


def score_both(y, pred, groups, task, eval_level, threshold=0.5):
    win = score(y, pred, groups, 'window', task, threshold)
    subj = score(y, pred, groups, 'subject', task, threshold)
    out = {f'{k}_win': v for k, v in win.items()}
    out.update({f'{k}_subj': v for k, v in subj.items()})
    out.update(subj if eval_level == 'subject' else win)
    return out


_PROVENANCE = None


def git_provenance():
    """{git_sha, git_dirty} for the tree this module lives in, resolved once.

    Every output row carries it, so the code that produced a row can be
    identified after the fact.
    """
    global _PROVENANCE
    if _PROVENANCE is None:
        repo = Path(__file__).resolve().parent
        try:
            sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                 capture_output=True, text=True, check=True).stdout.strip()
            dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                                        capture_output=True, text=True,
                                        check=True).stdout.strip())
            _PROVENANCE = {'git_sha': sha, 'git_dirty': dirty}
        except (subprocess.CalledProcessError, FileNotFoundError):
            _PROVENANCE = {'git_sha': None, 'git_dirty': None}
    return dict(_PROVENANCE)
