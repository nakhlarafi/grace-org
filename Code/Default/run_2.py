import torch
from torch import optim
from Dataset import SumDataset
import os
from tqdm import tqdm
from Model import *
import numpy as np
import time
from nltk import word_tokenize
import pickle
from ScheduledOptim import ScheduledOptim
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import random
import sys
import wandb
# wandb.init(project="codesum")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# math: 2986x977
# codec: 158x186
# Compress: 448x1114
# Gson 898x312
# Cli 497x293
# JacksonCore 289x144
NlLen_map = {"Time":3900, "Math":3000, "Lang":500, "Chart": 2350, "Mockito":1400, "Closure":5000, "Codec":500, "Compress":1000, "Gson":1000, "Cli":1000, "Jsoup":2000, "Csv":500, "JacksonCore":1000, 'JacksonXml':500, 'Collections':500}
CodeLen_map = {"Time":600, "Math":1000, "Lang":500, "Chart":5250, "Mockito":300, "Closure":10000, "Codec":500, "Compress":1500, "Gson":1000, "Cli":1000, "Jsoup":2000, "Csv":500, "JacksonCore":1000, 'JacksonXml':500, 'Collections':500}
args = dotdict({
    'NlLen':NlLen_map[sys.argv[2]],
    'CodeLen':CodeLen_map[sys.argv[2]],
    'SentenceLen':10,
    'batch_size':60,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':0,
    'lr':1e-3
})
os.environ['PYTHONHASHSEED'] = str(args.seed)

def save_model(model, dirs = "checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))

use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def test_model(test_set, model_path="checkpointcodeSearch/best_model.ckpt"):
    # Load the model
    model = NlEncoder(args)
    if use_cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))

    # Prepare the test set
    # Assuming test_set is already an instance of SumDataset or similar
    # If not, you need to instantiate it here

    # Set the model to evaluation mode
    model.eval()

    # Variable to store predictions
    predictions = []
    for testBatch in tqdm(test_set.Get_Train(args.batch_size)):
        testBatch = [gVar(x) for x in testBatch]
        with torch.no_grad():
            # Assuming your model returns predictions as its first output
            preds = model(*testBatch)[0]  # Modify as per your model's return values
            predictions.append(preds.cpu().numpy())  # Or handle as needed

    return predictions

if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    p = sys.argv[2]
    
    # Prepare your test dataset
    test_set = SumDataset(args, "test", p, testid=int(sys.argv[1]))

    # Test the model
    test_predictions = test_model(test_set)

    # Handle the test_predictions as needed
    # For example, you can save them to a file
    with open(f'{p}_test_predictions.pkl', 'wb') as file:
        pickle.dump(test_predictions, file)