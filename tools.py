import click
from scipy import linalg, mat, dot
from datetime import datetime
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import os
import shutil
import glob


def calc_similarity(v1, v2):
    return dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)


@click.group()
def cmd():
    pass


def make_preparation_dir(vec_file, vocab_file, dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    vec_fps = [open(os.path.join(dir_name, '{}.vec'.format(idx)), 'w') for idx in range(1, cpu_count() + 1)]
    for idx, row in enumerate(open(vec_file)):
        fp = vec_fps[idx % len(vec_fps)]
        fp.write('{}'.format(row))


def vec_dictionary_from_file_path(file_path):
    print(file_path, 'start')
    a = {row[0]: np.array(row[1:], dtype=float) for row in (row.strip().split() for row in open(file_path))}
    print(file_path, 'end')
    return a



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


def near_similarity_tokens(file_path, vec, cmp_func, border):
    for token, matrix in iter_vec_value(file_path):
        if vec.shape == matrix.shape:
            similarity = cmp_func(vec, matrix)
            if similarity >= border:
                yield token, similarity


@cmd.command()
@click.argument('vec_path', type=click.STRING)
@click.argument('dir_name', type=click.STRING)
def prepare(vec_path, vocab_path, dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    make_preparation_dir(vec_path, vocab_path, dir_name)


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('vocab_path', type=click.STRING)
@click.argument('preparation_dir', type=click.STRING)
def distance(vocab, vocab_path, preparation_dir):
    assert os.path.exists(preparation_dir)
    token = vocab2token(vocab, vocab_path)
    click.echo(token)
    now = datetime.now()
    vec_dict = VecDirDict(preparation_dir)
    print(vec_dict.get(token))
    print(datetime.now() - now)


def vocab2token(vocab, path):
    print(vocab, path)
    with open(path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            if vocab == row:
                return str(idx)
    return None


# @cmd.command()
# @click.argument('token', type=click.INT)
# @click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
# def token2vocab(token, path):
#     with open(path) as fp:
#         for idx, row in enumerate(map(lambda x: x.strip(), fp)):
#             if token == idx:
#                 return row
#     return None


if __name__ == '__main__':
    cmd()
