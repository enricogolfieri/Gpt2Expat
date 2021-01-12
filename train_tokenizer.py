from src.tokenise import BPE_token
from pathlib import Path
import CONFIG as conf

INPUT_DATASET_PATH = conf.DATASET_PATH_LANGUAGE
BACKUP_MODEL_PATH = conf.BACKUP_TOKENIZER_MODEL_PATH #hard disk or remote cloud
LOCAL_MODEL_PATH = conf.LOCAL_TOKENIZER_MODEL_PATH

print("loading files...")
# the folder 'text' contains all the files
paths = [str(x) for x in Path(INPUT_DATASET_PATH).glob("*.txt")]
tokenizer = BPE_token()

print("start training...")
# train the tokenizer model
tokenizer.bpe_train(paths)

print("saving..")
# saving the tokenized data both locally (ssd) in our specified folder 
tokenizer.save_tokenizer(BACKUP_MODEL_PATH)
tokenizer.save_tokenizer(LOCAL_MODEL_PATH)
print("done")