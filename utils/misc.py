#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import json
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
import os
import logging
from tqdm import trange

EMBED_DIM = 128
dtype = torch.cuda.FloatTensor \
    if torch.cuda.is_available() else torch.FloatTensor
DATA_KEY = "data"
VERSION_KEY = "version"
DOC_KEY = "document"
QAS_KEY = "qas"
ANS_KEY = "answers"
TXT_KEY = "text"  # the text part of the answer
ORIG_KEY = "origin"
ID_KEY = "id"
TITLE_KEY = "title"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
QUERY_KEY = "query"
CUI_KEY = "cui"
SEMTYPE_KEY = "sem_type"


def to_var(inputs, use_cuda, evaluate=False):
    if use_cuda:
        return Variable(torch.from_numpy(inputs).cuda(), volatile=evaluate)
    else:
        return Variable(torch.from_numpy(inputs), volatile=evaluate)


def to_vars(inputs, use_cuda, evaluate=False):
    return [to_var(inputs_, use_cuda, evaluate) for inputs_ in inputs]


def show_predicted_vs_ground_truth(probs, a, inv_dict):
    predicted_ans = list(map(
        lambda i: inv_dict[i], list(np.argmax(probs, axis=1))))
    true_ans = list(map(
        lambda i: inv_dict[i], list(a)))
    print(zip(predicted_ans, true_ans))


def count_candidates(probs, c, m_c):
    hits = 0
    predicted_ans = list(np.argmax(probs, axis=1))
    for i, x in enumerate(predicted_ans):
        for j, y in enumerate(c[i, :]):
            if x == y and m_c[i, j] > 0:
                hits += 1
                break
    return hits


def show_question(d, q, a, m_d, m_q, c, m_c, inv_dict):
    i = 0

    def inv_vocab(x):
        return inv_dict[x]
    print(list(map(inv_vocab, list(d[i, m_d[i] > 0, 0]))))
    print(list(map(inv_vocab, list(q[i, m_q[i] > 0, 0]))))
    print(list(map(inv_vocab, list(c[i, m_c[i] > 0]))))
    print(inv_vocab(a[i]))


def load_word2vec_embeddings(dictionary, vocab_embed_file):
    if vocab_embed_file is None:
        return None, EMBED_DIM

    fp = open(vocab_embed_file, encoding='utf-8')

    info = fp.readline().split()
    embed_dim = int(info[1])
    # vocab_embed: word --> vector
    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(
            list(map(float, line[1:])), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i, :] = vocab_embed[w]
            n += 1
    logging.info("{}/{} vocabs are initialized with word2vec embeddings."
                 .format(n, vocab_size))
    return W, embed_dim


def check_dir(*args, exit_function=False):
    """
    check the existence of directories
    Args:
    - args: (list) paths of directories
    - exit_function: (bool) action to take
    """
    for dir_ in args:
        if not os.path.exists(dir_):
            if not exit_function:
                os.makedirs(dir_)
            else:
                raise ValueError("{} does not exist!".format(dir_))


def prepare_input(d, q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i, :] = np.in1d(d[i, :, 0], q[i, :, 0])
    return f


def evaluate(model, data, use_cuda, inv_dict=None):
    acc = loss = n_examples = 0
    tr = trange(
        len(data),
        desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
        leave=False)
    if inv_dict is not None:
        preds = {}
        fn_to_i = {q[-1]: c for c, q in enumerate(data.questions)}
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, \
            tm, c, m_c, cl, fnames in data:
        bsize = dw.shape[0]
        n_examples += bsize
        dw, dt, qw, qt, a, m_dw, m_qw, tt, \
            tm, c, m_c, cl = to_vars([dw, dt, qw, qt, a, m_dw, m_qw, tt,
                                     tm, c, m_c, cl],
                                     use_cuda=use_cuda,
                                     evaluate=True)
        loss_, acc_, pred_ = model(dw, dt, qw, qt, a, m_dw, m_qw, tt,
                            tm, c, m_c, cl, fnames)
        _loss = float(loss_.cpu().data.numpy())
        _acc = float(acc_.cpu().data.numpy())
        if inv_dict is not None:
            # tidy up the answers
            for f, p in zip(fnames, pred_):
                cands = data.questions[fn_to_i[f]][3]
                pred_a = inv_dict[cands[int(p)][0]]
                assert pred_a.startswith("@entity")
                preds[os.path.basename(f).rsplit(".", 1)[0]] = pred_a[len("@entity"):].replace("_", " ")
        loss += _loss
        acc += _acc
        tr.set_description("loss: {:.3f}, acc: {:.3f}".
                           format(_loss, _acc / bsize))
        tr.update()
    tr.close()
    return loss / len(data), acc / n_examples, preds if inv_dict else None


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'))


def evaluate_clicr(test_file, preds_file, extended=False,
                   emb_file="/nas/corpora/accumulate/clicr/embeddings/b2257916-6a9f-11e7-aa74-901b0e5592c8/embeddings",
                   downcase=True):
    results = subprocess.check_output(
        "python3 ~/Apps/bmj_case_reports/evaluate.py -test_file {test_file} -prediction_file {preds_file} -embeddings_file {emb_file} {downcase} {extended}".format(
            test_file=test_file, preds_file=preds_file, emb_file=emb_file, downcase="-downcase" if downcase else "",
            extended="-extended" if extended else ""), shell=True)
    return results


def get_q_ids_clicr(fn):
    q_ids = set()
    dataset = load_json(fn)
    data = dataset[DATA_KEY]
    for datum in data:
        for qa in datum[DOC_KEY][QAS_KEY]:
            q_ids.add(qa[ID_KEY])

    return q_ids


def remove_missing_preds(fn, predictions):
    dataset = load_json(fn)
    new_dataset = intersect_on_ids(dataset, predictions)

    return new_dataset


def intersect_on_ids(dataset, predictions):
    """
    Reduce data to exclude all qa ids but those in  predictions.
    """
    new_data = []

    for datum in dataset[DATA_KEY]:
        qas = []
        for qa in datum[DOC_KEY][QAS_KEY]:
            if qa[ID_KEY] in predictions:
                qas.append(qa)
        if qas:
            new_doc = document_instance(datum[DOC_KEY][CONTEXT_KEY], datum[DOC_KEY][TITLE_KEY], qas)
            new_data.append(datum_instance(new_doc, datum[SOURCE_KEY]))

    return dataset_instance(dataset[VERSION_KEY], new_data)


def document_instance(context, title, qas):
    return {"context": context, "title": title, "qas": qas}


def dataset_instance(version, data):
    return {"version": version, "data": data}


def datum_instance(document, source):
        return {"document": document, "source": source}


def clicr_eval(path_test, path_preds, save_to):
    preds = load_json(path_preds)
    test_q_ids = get_q_ids_clicr(path_test)

    missing = test_q_ids - preds.keys()
    new_test = remove_missing_preds(path_test, preds.keys())
    test_file = "/tmp/reduced_test.json"
    save_json(new_test, test_file)
    results = evaluate_clicr(test_file, path_preds, extended=True, downcase=True)
    with open(save_to, "w") as outf:
        outf.write("\n{} predictions missing out of {}.".format(len(missing), len(test_q_ids)))
        outf.write("\nIgnoring missing predictions.")
        outf.write(results.decode())



