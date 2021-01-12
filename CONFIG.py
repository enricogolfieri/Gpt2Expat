import os

pretrained_model_name = 'egolnet'
id = 'small-base'

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATH = "C:/dataset/"
BACKUP_PATH = "D:/models/gpt2-italian/"
LOCAL_PATH = CURRENT_PATH
N_FILES = 10000 #-1 for all
BACKEND = 'tensorflow'

############################# DERIVED CONFIGURATION ##############################

BACKUP_TOKENIZER_MODEL_PATH = BACKUP_PATH + "{}_tokenizer".format(pretrained_model_name)
LOCAL_TOKENIZER_MODEL_PATH = LOCAL_PATH + "/tokenizers/{}/".format(pretrained_model_name)

#where to store the model?
BACKUP_MODEL_PATH = BACKUP_PATH + "{}/{}".format(pretrained_model_name,id)
LOCAL_MODEL_PATH = CURRENT_PATH + "/release/{}/{}/".format(pretrained_model_name,id)

#where do i get the data?
LANG = 'it'
DATASET_PATH_LANGUAGE = DATASET_PATH + "{}_corpus/".format(LANG)

#where do i store the checkpoints?
CHECKPOINT_PATH = CURRENT_PATH + "/checkpoints"
PRECOMPUTED_ENCODINGS = CURRENT_PATH + '/cache/' + "{}_corpus_{}_encodings".format(LANG, pretrained_model_name)

if pretrained_model_name == 'egolnet':

    BATCH_SIZE = 12
    BLOCK_SIZE = 128
    BUFFER_SIZE = 1000


if pretrained_model_name == 'geppetto':

    BATCH_SIZE = 32
    BLOCK_SIZE = 100
    BUFFER_SIZE = 1000
    MODEL_NAME = 'geppetto'


#in this case we assume you want to train from scratch
else:
    BATCH_SIZE = 12
    BLOCK_SIZE = 100
    BUFFER_SIZE = 1000



