# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# After we have encoded the whole string, we now move on to make a TensorFlow dataset, slicing the data into equal intervals, so that our model can learn. Here we use a block size of 100 (length of token in each example) and a batch size of 16. This is kept low else we can run it with ease on a RTX 2070 super GPU.
# %% [markdown]
# 1. Model Initialization
# Before the real magic begins, we need to make sure the artilleries are ready. Let us start with some initializations.

import sys
import os

#custom dependencies
import CONFIG as conf
import src.project as prj

print(sys.version)

# %% [markdown]
# We also create a single string from all our documents and tokenize it.
# %%


model, tokenizer = prj.load_project()

print("project loaded")

from src.util.data import load_and_encode

string_tokenized = load_and_encode(tokenizer, use_cache = True)


if conf.BACKEND == 'tensorflow':
    from src.tf import train_tf
    train_tf(model, tokenizer, string_tokenized, num_epoch=1)
elif conf.BACKEND == 'torch':
    from src.torch import train_torch
    train_torch(model, tokenizer, string_tokenized, num_epoch=1)
else:
    print(" what is this? {} please choose one backend between torch or tensorflow".format(conf.BACKEND))

model_to_save = model.module if hasattr(model, 'module') else model

# save
prj.save_project( tokenizer, model_to_save, save_locally=True)

