import click
from scipy import linalg, mat, dot
from datetime import datetime
import numpy as np
import time

def similarity(v1, v2):
    return dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)


@click.group()
def cmd():
    pass


@cmd.command()
@click.argument('vocab', type=click.STRING)
@click.argument('vec_path', type=click.STRING)
@click.argument('vocab_path', type=click.STRING)
def distance(vocab, vec_path, vocab_path):
    token1 = vocab2token('宗教', vocab_path)
    token2 = vocab2token('戦争', vocab_path)
    # token3 = vocab2token('性欲', vocab_path)
    token2vec = {}
    token2vocab = {}
    with open(vocab_path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            token2vocab[str(idx)] = row
    print('----token2vocab')
    with open(vec_path) as fp:
        now = datetime.now()
        for idx, row in enumerate(map(lambda x: x.strip().split(), fp)):
            vec = np.array(row[1:], dtype=float)
            token2vec[row[0]] = vec
        print(datetime.now() - now)
    print('---token2vec')
    vec = token2vec[token1] - token2vec[token2]
    for t, v in token2vec.items():
        if not vec.shape == v.shape:
            print('continue')
            continue
        s = similarity(vec, v)
        if s > 0.6:
            print(vocab, token2vocab[t], s)


def vocab2token(vocab, path):
    print(vocab, path)
    with open(path) as fp:
        for idx, row in enumerate(map(lambda x: x.strip(), fp)):
            if vocab == row:
                click.echo(idx)
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
