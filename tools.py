import click
from scipy import linalg, dot
from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import shutil
import glob
from itertools import chain


def calc_similarity(v1, v2):
    return dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)


def similarity_tokens(args):
    similarity_func, vec, vec_dictionary, border = args
    values = ((t, similarity_func(vec, v)) for t, v in vec_dictionary.items())
    return list(filter(lambda ts: ts[1] >= border, values))


@click.group()
def cmd():
    pass


def make_preparation_dir(vec_file, dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    vec_fps = [open(os.path.join(dir_name, '{}.vec'.format(idx)), 'w') for idx in range(1, cpu_count() + 1)]
    with open(vec_file) as fp:
        fp.readline()
        for idx, row in enumerate(fp):
            fp = vec_fps[idx % len(vec_fps)]
            fp.write('{}'.format(row))


def vec_dictionary_from_file_path(file_path):
    return {row[0]: np.array(row[1:], dtype=float) for row in (row.strip().split() for row in open(file_path))}


class VecDirDict:
    def __init__(self, dir_name):
        super().__init__()
        self.dir_name = dir_name
        self.dicts = Pool().map(
            vec_dictionary_from_file_path,
            (os.path.join(self.dir_name, x) for x in glob.glob1(self.dir_name, '*.vec'))
        )

    def get(self, key):
        for dictionary in self.dicts:
            if key in dictionary:
                return dictionary[key]
        return None

    def similarity_tokens(self, vec, border):
        return chain.from_iterable(Pool().map(
            similarity_tokens,
            ((calc_similarity, vec, d, border) for d in self.dicts)
        ))


@cmd.command()
@click.argument('vec_path', type=click.STRING)
@click.argument('dir_name', type=click.STRING, default='preparation_dir')
def prepare(vec_path, dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    make_preparation_dir(vec_path, dir_name)


@cmd.command()
@click.argument('vec_path', type=click.STRING)
@click.argument('dist_vec', type=click.STRING)
def prepare_for_ngt(vec_path, dist_vec):
    dist = open(dist_vec, 'w')
    vec_file = open(vec_path)
    vec_file.readline()
    for key, row in (row.strip().split(maxsplit=1) for row in vec_file):
        vec = np.array(row.split(), dtype='float32')
        vec /= np.linalg.norm(vec)
        row = list(vec)
        row = ' '.join(map(str, row))
        dist.write('{} {}\n'.format(row, key))


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('vocab_path', type=click.STRING)
@click.argument('preparation_dir', type=click.STRING, default='preparation_dir')
@click.argument('border', type=click.FLOAT, default=0.65)
def distance(vocab, vocab_path, preparation_dir, border):
    assert os.path.exists(preparation_dir)
    token = vocab2token(vocab, vocab_path)
    click.echo(token)
    now = datetime.now()
    vec_dict = VecDirDict(preparation_dir)
    print(datetime.now() - now)

    now = datetime.now()
    values = vec_dict.similarity_tokens(vec_dict.get(token), border)
    print(datetime.now() - now)
    values = sorted(values, key=lambda ts: ts[1], reverse=True)
    print(values)
    t2v = tokens2vocab([t for t, s in values], vocab_path)
    for t, s in values:
        click.echo('{} {}'.format(t2v.get(t), s))


def vocab2token(vocab, path):
    print(vocab, path)
    with open(path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            if vocab == row:
                return str(idx)
    return None


def tokens2vocab(tokens, path):
    result = {}
    print(tokens, path)
    with open(path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            idx = str(idx)
            if idx in tokens:
                result[idx] = row
                tokens.remove(idx)
                if not tokens:
                    break
    return result


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def v2t(vocab, path):
    click.echo(vocab2token(vocab, path))


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('vocab_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('vec_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def v2v(vocab, vocab_path, vec_path):
    token = vocab2token(vocab, vocab_path)
    click.echo(token)
    vec_file = open(vec_path)
    vec_file.readline()
    for vec, key in (row.strip().rsplit(maxsplit=1) for row in vec_file):
        if token == key:
            print(key, vec)
            return


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def vocab2vec(vocab, path):
    token2vocab(vocab)
    print(vocab, path)
    with open(path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            if vocab == row:
                return str(idx)
    return None


@cmd.command()
@click.argument('token', type=click.STRING)
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def token2vocab(token, path):
    print(token, path)
    print(tokens2vocab([token], path))


if __name__ == '__main__':
    cmd()
