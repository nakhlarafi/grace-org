import pickle
import sys
file = sys.argv[1]
with open(file, 'rb') as f:
    data = pickle.load(f)
    print(data)