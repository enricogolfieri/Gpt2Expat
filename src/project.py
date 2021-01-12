import CONFIG as conf

import CONFIG as conf
import os

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoTokenizer, GPT2LMHeadModel, GPT2Config


def cached_tokenizer(function):
    def wrapper(*args, **kwargs):
        if os.path.exists(conf.LOCAL_TOKENIZER_MODEL_PATH):
            tokenizer = GPT2Tokenizer.from_pretrained(conf.LOCAL_TOKENIZER_MODEL_PATH)
            print("loaded local tokenizer from {}".format(conf.LOCAL_TOKENIZER_MODEL_PATH))
            return tokenizer
        elif os.path.exists(conf.BACKUP_TOKENIZER_MODEL_PATH):
            tokenizer = GPT2Tokenizer.from_pretrained(conf.BACKUP_TOKENIZER_MODEL_PATH)
            print("loaded backup tokenizer from {}".format(conf.BACKUP_TOKENIZER_MODEL_PATH))
            return tokenizer
        else:
            return function(*args, **kwargs)

    return wrapper


def cached_model(function):
    def wrapper(*args, **kwargs):
        if os.path.exists(conf.LOCAL_MODEL_PATH):
            m = TFGPT2LMHeadModel.from_pretrained(conf.LOCAL_MODEL_PATH)
            print("loaded local tokenizer from {}".format(conf.LOCAL_MODEL_PATH))
            return m
        elif os.path.exists(conf.BACKUP_MODEL_PATH):
            m = TFGPT2LMHeadModel.from_pretrained(conf.BACKUP_MODEL_PATH)
            print("loaded backup tokenizer from {}".format(conf.BACKUP_MODEL_PATH))
            return m
        else:
            return function(*args, **kwargs)

    return wrapper


@cached_tokenizer
def geppetto_tokenizer():
    return AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")


@cached_model
def geppetto_model():
    m = GPT2LMHeadModel.from_pretrained("LorenzoDeMattei/GePpeTto")
    if conf.BACKEND == 'tensorflow':
        import warnings
        warnings.warn('backend set to tensorflow but geppetto can be only trained with torch, MODE set on torch')
        conf.BACKEND = 'torch'
    return m


@cached_tokenizer
def egolnet_tokenizer():
    return GPT2Tokenizer.from_pretrained(conf.CURRENT_PATH + "/tokenizers/egolnet_tokenizer")


@cached_model
def egolnet_model():
    return TFGPT2LMHeadModel.from_pretrained("gpt2")


@cached_tokenizer
def load_tokenizer():
    raise Exception("no tokenizer found, impossible to proceed forward")


@cached_model
def load_model(tokenizer):
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })

    # creating the configurations from which the model can be made
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # creating the model
    return TFGPT2LMHeadModel(config)


def torch_to_keras(gpt2_torch_model):
    '''not working'''
    import onnx
    pytorch_model = conf.CURRENT_PATH + '/cache/temp'
    keras_output = conf.CURRENT_PATH + '/cache/temp.hdf5'

    gpt2_torch_model.save_pretrained(pytorch_model)

    onnx.convert(pytorch_model, keras_output)
    model = TFGPT2LMHeadModel.from_pretrained(keras_output)
    return model


def cache_project_model(model) -> str:
    import os

    path = conf.CURRENT_PATH + "/cache/" + type(model).__name__

    if not os.path.exists(path):
        os.mkdir(path)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)

    return path + "/tf_model.h5"


def save_project( tokenizer, model_to_save, save_locally=True):
    # save model and model configs
    model_to_save.save_pretrained(conf.BACKUP_MODEL_PATH)
    # save tokenizer
    tokenizer.save_pretrained(conf.BACKUP_TOKENIZER_MODEL_PATH)

    if save_locally:
        # save model and model configs
        model_to_save.save_pretrained(conf.LOCAL_MODEL_PATH)
        # save tokenizer
        tokenizer.save_pretrained(conf.LOCAL_TOKENIZER_MODEL_PATH)


def load_project():
    '''
    Load both tokenizer and model
    Tokenizer: first search locally, otherwise in backup folder
    Model: first search locally, otherwise in backup folder
    in case nothing is found , a new training will start from scratch

    if no tokenizer is found, project cannot be loaded. please refer to train_tokenizer to train a new one.
    :param conf:
    :return: tuple Keras model and Tokenizer
    '''

    # loading tokenizer from the saved model path
    if conf.pretrained_model_name == 'geppetto':
        model = geppetto_model()
        tokenizer = geppetto_tokenizer()

    elif conf.pretrained_model_name == 'egolnet':
        model = egolnet_model()
        tokenizer = egolnet_tokenizer()
    else:
        tokenizer = load_tokenizer()
        model = load_model(tokenizer)

    return model, tokenizer
