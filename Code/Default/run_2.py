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

def test(p='Math', t=5):
    # Load the datasets
    dev_set = SumDataset(args, "test", p, testid=t)
    data = pickle.load(open(p + '.pkl', 'rb'))

    # Initialize and load the model
    model = NlEncoder(args)
    if use_cuda:
        model = model.cuda()
    load_model(model, dirs="checkpointcodeSearch")

    # Set the model to evaluation mode
    model.eval()

    # Container for results
    test_results = []
    score_dict = {}

    # Testing loop
    for k, devBatch in tqdm(enumerate(dev_set.Get_Train(args.batch_size))):
        devBatch = [gVar(x) for x in devBatch]  # Prepare batch data
        with torch.no_grad():
            l, pre, _ = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3], devBatch[4], devBatch[5], devBatch[6], devBatch[7])
            resmask = torch.eq(devBatch[0], 2)
            s = -pre  # Adjust based on how your model outputs predictions
            s = s.masked_fill(resmask == 0, 1e9)
            pred = s.argsort(dim=-1)
            pred = pred.data.cpu().numpy()

            for k in range(len(pred)): 
                datat = data[dev_set.ids[k]]
                maxn = 1e9
                lst = pred[k].tolist()[:resmask.sum(dim=-1)[k].item()]
                for pos in lst:
                    score_dict[pos] = s[k, pos].item()
                for x in datat['ans']:
                    i = lst.index(x)
                    maxn = min(maxn, i)
                test_results.append(maxn)

    return test_results, score_dict



if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    np.set_printoptions(threshold=sys.maxsize)
    res = {}    
    p = sys.argv[2]
    res[int(sys.argv[1])] = train(int(sys.argv[1]), p)
    open('%sres%d_%d_%s_%s.pkl'%(p, int(sys.argv[1]), args.seed, args.lr, args.batch_size), 'wb').write(pickle.dumps(res))


