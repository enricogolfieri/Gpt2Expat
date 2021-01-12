# %%

from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead

import time
import os
import CONFIG as conf


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def load_and_merge(paths_list):
    '''
    like load and merge but just for the first 100 files, useful to prototype the training
    :param paths_list:
    :param tokenizer:
    :param single_string: pre-initialized empty string, we do not want to return the result it's too big.
    :return: void. single string is filled
    '''
    if conf.N_FILES == -1:
        return ''.join([open(filename, "r", encoding='utf-8').read() for filename in paths_list])
    else:
        return ''.join([open(filename, "r", encoding='utf-8').read() for filename in paths_list[:conf.N_FILES]])


def load_and_encode(tokenizer, use_cache=True):
    import numpy as np
    from pathlib import Path

    if os.path.exists(conf.PRECOMPUTED_ENCODINGS + '.npy') and use_cache:
        file = open(conf.PRECOMPUTED_ENCODINGS + '.npy', 'rb')
        string_tokenized = np.load(file, allow_pickle=True)
        print("loaded precomputed encodings at {}".format(conf.PRECOMPUTED_ENCODINGS))
        return string_tokenized
    else:
        paths = [str(x) for x in Path(conf.DATASET_PATH_LANGUAGE).glob("*.txt")]
        print("total file to read {} ".format(len(paths)))

        string_tokenized = np.array([])

        print("running load and merge")
        single_string = ''
        single_string = load_and_merge(paths)

        print("loaded into one string, len: ", len(single_string))

        print("running encoding")
        string_tokenized = tokenizer.encode(single_string)

        print("caching all in one file..")
        with open(conf.PRECOMPUTED_ENCODINGS + '.npy', 'wb') as output:
            np.save(output, string_tokenized, allow_pickle=True, fix_imports=True)
            output.flush()

        return string_tokenized
