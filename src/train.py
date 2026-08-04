"""
Trains the 3 MCQ models, compares them, and pushes model B to the hub.
Project T22026 - 24f2005363

  model A : tfidf features + a plain net
  model B : MiniLM sentence embeddings + small head
  model C : MPNet embeddings + wider head

run:
    python src/train.py --data-dir data --out-dir outputs
    python src/train.py --no-wandb --no-push --quick     # fast local check
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (CHOICES, EMB_NAME_B, EMB_NAME_C, HF_REPO_SUFFIX, WB_PROJECT,
                   NetA, NetB, NetC, Run, SimpleData, add_text, embed_texts,
                   get_device, get_map3, hf_login, load_data, make_split,
                   predict_probs, run_epoch, save_json, score_all, set_seed,
                   write_submission)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=None, help='folder with train.csv / test.csv')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--epochs-a', type=int, default=22)
    p.add_argument('--epochs-b', type=int, default=18)
    p.add_argument('--epochs-c', type=int, default=18)
    p.add_argument('--max-features', type=int, default=8000, help='tfidf vocab cap')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--no-push', action='store_true', help='skip the huggingface upload')
    p.add_argument('--skip-c', action='store_true', help='skip MPNet, it is the slow one')
    p.add_argument('--quick', action='store_true', help='2 epochs each, for testing the pipeline')
    return p.parse_args()


def train_one(net, feat_tr, y_tr, feat_va, y_va, epochs, lr, batch_size,
              pick_on, ckpt, run, dev, tag):
    """
    One training loop shared by all 3 models. pick_on says which val metric
    decides the saved checkpoint - 'f1' or 'map3'.
    """
    loader_tr = DataLoader(SimpleData(feat_tr, y_tr), batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(SimpleData(feat_va, y_va), batch_size=batch_size)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    for ep in range(epochs):
        tl, _, _, _ = run_epoch(net, loader_tr, opt, loss_fn, True, dev)
        vl, pred_v, true_v, prob_v = run_epoch(net, loader_va, opt, loss_fn, False, dev)

        m = score_all(true_v, prob_v)
        cur = m[pick_on]
        if cur > best:
            best = cur
            torch.save(net.state_dict(), ckpt)

        run.log({'epoch': ep + 1, 'train_loss': tl, 'val_loss': vl,
                 'val_acc': m['acc'], 'val_f1': m['f1'], 'val_map3': m['map3']})
        print(tag, 'epoch', ep + 1, '| train loss', round(tl, 4), '| val loss', round(vl, 4),
              '| acc', round(m['acc'], 4), '| f1', round(m['f1'], 4), '| map3', round(m['map3'], 4))

    net.load_state_dict(torch.load(ckpt, map_location=dev))
    print(tag, 'best val', pick_on, '=', round(best, 4))
    run.set_summary('best_val_' + pick_on, best)
    return net, best


def finish_model(name, net, feat_va, y_va, feat_te, df_sub, dev, out_dir,
                 batch_size, table_rows, val_probs, test_probs):
    """Score on val, save probs, write the per model submission."""
    va_prob = predict_probs(net, feat_va, dev, batch_size)
    te_prob = predict_probs(net, feat_te, dev, batch_size)

    m = score_all(y_va, va_prob)
    val_probs[name] = va_prob
    test_probs[name] = te_prob
    table_rows.append([name, m['acc'], m['f1'], m['map3']])

    np.save(os.path.join(out_dir, 'val_prob_%s.npy' % name), va_prob)
    np.save(os.path.join(out_dir, 'test_prob_%s.npy' % name), te_prob)
    write_submission(te_prob, df_sub, os.path.join(out_dir, 'sub_%s.csv' % name))

    print(name, '-> acc', round(m['acc'], 4), 'f1', round(m['f1'], 4), 'map3', round(m['map3'], 4))
    return m


def push_to_hub(hf_user, out_dir, table, top_name, files):
    """Upload model B and the comparison files to a hub repo."""
    from huggingface_hub import HfApi, create_repo

    repo_name = hf_user + '/' + HF_REPO_SUFFIX
    create_repo(repo_name, exist_ok=True)
    hub = HfApi()

    b_row = table[table['model'] == 'model_b'].iloc[0]
    note = (
        "# MCQ predictor - model_b\n\n"
        "MiniLM sentence-embedding model pushed for the T22026 project.\n"
        "val accuracy: " + str(round(b_row['val_acc'], 4)) + "\n"
        "val map3: " + str(round(b_row['val_map3'], 4)) + "\n\n"
        "best model in this run on val_map3 was " + str(top_name) +
        " (see score_table.csv for the full comparison).\n"
        "wandb project: " + WB_PROJECT + "\n"
    )
    readme = os.path.join(out_dir, 'README.md')
    with open(readme, 'w') as f:
        f.write(note)

    for p in files + [readme]:
        if os.path.exists(p):
            hub.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=repo_name)
            print('uploaded', p)

    print('done, see https://huggingface.co/' + repo_name)
    return repo_name


def main():
    args = parse_args()
    if args.quick:
        args.epochs_a = args.epochs_b = args.epochs_c = 2

    set_seed(args.seed)
    dev = get_device()
    print('using device', dev)
    os.makedirs(args.out_dir, exist_ok=True)

    hf_user = None if args.no_push else hf_login()
    group = hf_user or 'local'
    use_wb = not args.no_wandb

    # ---- data
    df_train, df_test, df_sub = load_data(args.data_dir)
    print('answer counts:')
    print(df_train['answer'].value_counts())
    print('repeated prompts:', len(df_train) - df_train['prompt'].nunique())

    df_tr, df_va, y_tr, y_va, label_enc = make_split(df_train, args.seed)
    df_tr, df_va, df_test = add_text(df_tr), add_text(df_va), add_text(df_test)

    text_tr = df_tr['text'].tolist()
    text_va = df_va['text'].tolist()
    text_te = df_test['text'].tolist()

    save_json({'classes': list(label_enc.classes_), 'seed': args.seed},
              os.path.join(args.out_dir, 'label_meta.json'))

    # answer balance plot
    cnt = df_train['answer'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(cnt.index, cnt.values, color='teal')
    plt.title('how many of each answer letter')
    plt.xlabel('letter')
    plt.ylabel('count')
    plt.savefig(os.path.join(args.out_dir, 'answer_count.png'))
    plt.close()

    table_rows, val_probs, test_probs = [], {}, {}

    # ---- model A, tfidf
    print('\n=== model A - tfidf + net ===')
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    fa_tr = vec.fit_transform(text_tr).toarray()
    fa_va = vec.transform(text_va).toarray()
    fa_te = vec.transform(text_te).toarray()
    print('tfidf feature size', fa_tr.shape)

    run_a = Run(use_wb, project=WB_PROJECT, name='model-a-' + group, group=group, job_type='train',
                config={'model': 'model_a', 'architecture': 'tfidf+mlp',
                        'epochs': args.epochs_a, 'lr': 0.001, 'batch_size': 64})
    net_a = NetA(fa_tr.shape[1]).to(dev)
    net_a, _ = train_one(net_a, fa_tr, y_tr, fa_va, y_va, args.epochs_a, 0.001, 64,
                         'f1', os.path.join(args.out_dir, 'net_a.pt'), run_a, dev, 'A')
    run_a.finish()
    finish_model('model_a', net_a, fa_va, y_va, fa_te, df_sub, dev, args.out_dir,
                 64, table_rows, val_probs, test_probs)

    # ---- model B, MiniLM
    print('\n=== model B - MiniLM embeddings ===')
    fb_tr, fb_va, fb_te = embed_texts(EMB_NAME_B, [text_tr, text_va, text_te], dev, 32)
    print('embedding size', fb_tr.shape[1])

    run_b = Run(use_wb, project=WB_PROJECT, name='model-b-' + group, group=group, job_type='train',
                config={'model': 'model_b', 'architecture': 'MiniLM-embeddings+mlp',
                        'embedding_model': EMB_NAME_B, 'epochs': args.epochs_b,
                        'lr': 0.001, 'batch_size': 32})
    net_b = NetB(fb_tr.shape[1]).to(dev)
    net_b, _ = train_one(net_b, fb_tr, y_tr, fb_va, y_va, args.epochs_b, 0.001, 32,
                         'map3', os.path.join(args.out_dir, 'net_b.pt'), run_b, dev, 'B')
    run_b.finish()
    finish_model('model_b', net_b, fb_va, y_va, fb_te, df_sub, dev, args.out_dir,
                 32, table_rows, val_probs, test_probs)
    save_json({'emb_name': EMB_NAME_B, 'in_size': int(fb_tr.shape[1])},
              os.path.join(args.out_dir, 'model_b_meta.json'))

    # ---- model C, MPNet
    if not args.skip_c:
        print('\n=== model C - MPNet embeddings ===')
        fc_tr, fc_va, fc_te = embed_texts(EMB_NAME_C, [text_tr, text_va, text_te], dev, 16)
        print('embedding size', fc_tr.shape[1])

        run_c = Run(use_wb, project=WB_PROJECT, name='model-c-' + group, group=group, job_type='train',
                    config={'model': 'model_c', 'architecture': 'MPNet-embeddings+mlp',
                            'embedding_model': EMB_NAME_C, 'epochs': args.epochs_c,
                            'lr': 0.0007, 'batch_size': 32})
        net_c = NetC(fc_tr.shape[1]).to(dev)
        net_c, _ = train_one(net_c, fc_tr, y_tr, fc_va, y_va, args.epochs_c, 0.0007, 32,
                             'f1', os.path.join(args.out_dir, 'net_c.pt'), run_c, dev, 'C')
        run_c.finish()
        finish_model('model_c', net_c, fc_va, y_va, fc_te, df_sub, dev, args.out_dir,
                     32, table_rows, val_probs, test_probs)

    # ---- compare
    table = pd.DataFrame(table_rows, columns=['model', 'val_acc', 'val_f1', 'val_map3'])
    table = table.sort_values('val_map3', ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(args.out_dir, 'score_table.csv'), index=False)
    print('\n', table, sep='')

    plt.figure(figsize=(7, 4))
    plt.bar(table['model'], table['val_map3'], color='indianred')
    plt.title('map3 score by model')
    plt.ylabel('map3')
    plt.savefig(os.path.join(args.out_dir, 'compare_plot.png'))
    plt.close()

    top_name = table.iloc[0]['model']
    top_map3 = float(table.iloc[0]['val_map3'])
    print('winner is', top_name, 'with map3 of', round(top_map3, 4))
    save_json({'winner': top_name, 'map3': top_map3}, os.path.join(args.out_dir, 'winner.json'))

    # ---- push model B, that is what the brief asks for
    repo_name = None
    if hf_user and not args.no_push:
        files = [os.path.join(args.out_dir, f) for f in
                 ['net_b.pt', 'model_b_meta.json', 'score_table.csv', 'winner.json', 'compare_plot.png']]
        repo_name = push_to_hub(hf_user, args.out_dir, table, top_name, files)
        save_json({'repo': repo_name}, os.path.join(args.out_dir, 'hub_repo.json'))
    else:
        print('skipping hub upload')

    print('\n' + '-' * 50)
    print('SUMMARY')
    print('-' * 50)
    print(table)
    print('winner (best val_map3):', top_name)
    print('model pushed to hub: model_b ->', repo_name)
    print('wandb project:', WB_PROJECT, '(runs grouped under', group + ')')
    print('files in', args.out_dir + ':')
    for f in sorted(os.listdir(args.out_dir)):
        print(' -', f)


if __name__ == '__main__':
    main()
