"""
Inference with model B, weights pulled straight from the Hugging Face repo.
Project T22026 - 24f2005363

This is the proof that the upload actually works - nothing here reuses the
in-memory embeddings from training, the text gets encoded again from scratch
and the net is a fresh NetB with hub weights loaded into it.

run:
    python src/inference.py --repo-id yourname/mcq-answer-predictor-t22026
    python src/inference.py --weights outputs/net_b.pt --no-wandb   # local weights
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (EMB_NAME_B, HF_REPO_SUFFIX, WB_PROJECT, NetB, Run, add_text,
                   embed_texts, get_device, hf_login, load_data, load_json,
                   make_split, predict_probs, save_json, score_all, set_seed,
                   write_submission)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=None)
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--repo-id', default=None, help='hub repo, else read from outputs/hub_repo.json')
    p.add_argument('--weights', default=None, help='use a local .pt instead of the hub')
    p.add_argument('--filename', default='net_b.pt', help='weight file inside the repo')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--no-val', action='store_true', help='predict test only, skip val scoring')
    return p.parse_args()


def resolve_weights(args, out_dir):
    """Local file if given, otherwise download from the hub."""
    if args.weights:
        print('using local weights:', args.weights)
        return args.weights, 'local'

    repo = args.repo_id
    if not repo:
        meta = os.path.join(out_dir, 'hub_repo.json')
        if os.path.exists(meta):
            repo = load_json(meta)['repo']
    if not repo:
        user = hf_login()
        if user:
            repo = user + '/' + HF_REPO_SUFFIX
    if not repo:
        raise SystemExit('no repo id and no --weights given, cannot load a model')

    hf_login()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=args.filename)
    print('downloaded weights from hub to:', path)
    return path, repo


def make_plots(y_va, prob, classes, out_path):
    cm = confusion_matrix(y_va, prob.argmax(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(classes)
    axes[0].set_xlabel('predicted')
    axes[0].set_ylabel('true')
    axes[0].set_title('confusion matrix (hub model B)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=8)

    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    axes[1].bar(list(classes)[:len(per_class)], per_class, color='slateblue')
    axes[1].set_title('per-class accuracy (hub model B)')
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('wrote', out_path)


def main():
    import torch

    args = parse_args()
    set_seed(args.seed)
    dev = get_device()
    print('using device', dev)
    os.makedirs(args.out_dir, exist_ok=True)

    weights_path, source = resolve_weights(args, args.out_dir)

    # same split as training, same seed, so val stays honest
    df_train, df_test, df_sub = load_data(args.data_dir)
    df_tr, df_va, y_tr, y_va, label_enc = make_split(df_train, args.seed)
    df_va, df_test = add_text(df_va), add_text(df_test)
    text_va = df_va['text'].tolist()
    text_te = df_test['text'].tolist()

    # fresh embeddings, no reuse from training
    emb_name = EMB_NAME_B
    meta_path = os.path.join(args.out_dir, 'model_b_meta.json')
    if os.path.exists(meta_path):
        emb_name = load_json(meta_path).get('emb_name', EMB_NAME_B)

    if args.no_val:
        (emb_te,) = embed_texts(emb_name, [text_te], dev, args.batch_size)
        emb_va = None
    else:
        emb_va, emb_te = embed_texts(emb_name, [text_va, text_te], dev, args.batch_size)

    in_size = emb_te.shape[1]
    net = NetB(in_size).to(dev)
    state = torch.load(weights_path, map_location=dev)
    net.load_state_dict(state)
    net.eval()
    print('loaded weights into a fresh NetB, input size', in_size)

    te_prob = predict_probs(net, emb_te, dev, args.batch_size)
    sub_path = os.path.join(args.out_dir, 'submission_hub_model_b.csv')
    write_submission(te_prob, df_sub, sub_path)
    np.save(os.path.join(args.out_dir, 'infer_test_prob.npy'), te_prob)

    res = None
    if not args.no_val:
        va_prob = predict_probs(net, emb_va, dev, args.batch_size)
        res = score_all(y_va, va_prob)
        print('hub model B inference -> acc', round(res['acc'], 4),
              'f1', round(res['f1'], 4), 'map3', round(res['map3'], 4))

        cm_path = os.path.join(args.out_dir, 'inference_confusion.png')
        make_plots(y_va, va_prob, label_enc.classes_, cm_path)
        save_json({'source': source, 'weights': str(weights_path), **res},
                  os.path.join(args.out_dir, 'infer_scores.json'))

        run = Run(not args.no_wandb, project=WB_PROJECT,
                  name='inference-hub-model-b', group='inference', job_type='inference',
                  config={'model': 'model_b', 'source': source, 'repo': str(source)})
        run.log({'infer_acc': res['acc'], 'infer_f1': res['f1'], 'infer_map3': res['map3']})
        run.log_image('confusion_matrix', cm_path)
        run.set_summary('infer_acc', res['acc'])
        run.set_summary('infer_map3', res['map3'])
        run.finish()

    print('\n' + '-' * 50)
    print('INFERENCE SUMMARY (model pulled from ' + str(source) + ')')
    print('-' * 50)
    if res:
        print('acc:', round(res['acc'], 4))
        print('f1 (macro):', round(res['f1'], 4))
        print('map3:', round(res['map3'], 4))
    print('submission written to', sub_path)


if __name__ == '__main__':
    main()
