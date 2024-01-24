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

def test(t=5, p='Math', model_dir="checkpointcodeSearch"):
    # Set up environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed + t)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load test dataset
    test_set = SumDataset(args, "val", p, testid=t)
    data = pickle.load(open(p + '.pkl', 'rb'))
    args.Code_Vocsize = len(test_set.Code_Voc)
    args.Nl_Vocsize = len(test_set.Nl_Voc)
    args.Vocsize = len(test_set.Char_Voc)

    # Initialize model and load weights
    model = NlEncoder(args)
    model.load_state_dict(torch.load(model_dir + '/best_model.ckpt'))
    if use_cuda:
        model = model.cuda()
    model.eval()

    # Testing loop
    brest = []
    bans = []
    batchn = []
    each_epoch_pred = {}
    cumulative_test_time = 0

    for k, testBatch in tqdm(enumerate(test_set.Get_Train(args.batch_size))):
        test_start_time = time.time()
        testBatch = [gVar(x) for x in testBatch]

        with torch.no_grad():
            l, pre, _ = model(*testBatch)
            resmask = torch.eq(testBatch[0], 2)
            s = -pre
            s = s.masked_fill(resmask == 0, 1e9)
            pred = s.argsort(dim=-1)
            pred = pred.data.cpu().numpy()

            score_dict = {}
            score2 = []
            for idx in range(len(pred)):
                datat = data[test_set.ids[idx]]
                maxn = 1e9
                lst = pred[idx].tolist()[:resmask.sum(dim=-1)[idx].item()]
                for pos in lst:
                    score_dict[pos] = s[idx, pos].item()
                for x in datat['ans']:
                    i = lst.index(x)
                    maxn = min(maxn, i)
                score2.append(maxn)

            cumulative_test_time += time.time() - test_start_time
            each_epoch_pred[k] = lst
            each_epoch_pred[str(k) + '_pred'] = score_dict
            brest.append(score2)
            if score2[0] == 0:
                batchn.append(k)
            bans.append(lst)
            print(each_epoch_pred)

    print(f"Total Testing Time: {cumulative_test_time}")

    return brest, bans, batchn, each_epoch_pred

# Main execution
if __name__ == "__main__":
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    p = sys.argv[2]
    res = {}
    print('-'*50)
    print(int(sys.argv[1]))
    res[int(sys.argv[1])] = test(int(sys.argv[1]), p)
    # print(res)
    open('%sres%d_%d_%s_%s.pkl'%(p, int(sys.argv[1]), args.seed, args.lr, args.batch_size), 'wb').write(pickle.dumps(res))
  
