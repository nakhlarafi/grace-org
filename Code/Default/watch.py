import pickle
import sys, os
import numpy as np

pr = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])
dmap = {
    'Math':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55, 54: 56, 55: 57, 56: 58, 57: 59, 58: 60, 59: 61, 60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68, 67: 69, 68: 70, 69: 71, 70: 72, 71: 73, 72: 74, 73: 75, 74: 76, 75: 77, 76: 78, 77: 79, 78: 80, 79: 81, 80: 82, 81: 83, 82: 84, 83: 85, 84: 86, 85: 87, 86: 88, 87: 89, 88: 90, 89: 91, 90: 92, 91: 93, 92: 94, 93: 95, 94: 96, 95: 97, 96: 98, 97: 99, 98: 100, 99: 101, 100: 102, 101: 103, 102: 105, 103: 106},
    'Lang': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57, 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    # 'Lang': {0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57, 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    'Chart':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 25, 24: 26},
    'Time':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},
    'Mockito':{0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38},
    'Closure':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58, 56: 59, 57: 60, 58: 61, 59: 64, 60: 65, 61: 66, 62: 67, 63: 68, 64: 69, 65: 70, 66: 71, 67: 72, 68: 73, 69: 74, 70: 75, 71: 76, 72: 77, 73: 78, 74: 79, 75: 80, 76: 81, 77: 82, 78: 83, 79: 84, 80: 85, 81: 86, 82: 87, 83: 88, 84: 89, 85: 90, 86: 91, 87: 92, 88: 94, 89: 95, 90: 96, 91: 97, 92: 98, 93: 99, 94: 100, 95: 101, 96: 102, 97: 103, 98: 104, 99: 105, 100: 106, 101: 108, 102: 109, 103: 110, 104: 111, 105: 112, 106: 114, 107: 116, 108: 117, 109: 118, 110: 119, 111: 120, 112: 121, 113: 122, 114: 123, 115: 124, 116: 125, 117: 126, 118: 127, 119: 128, 120: 129, 121: 130, 122: 131, 123: 133, 124: 134, 125: 135, 126: 136, 127: 137, 128: 138, 129: 139, 130: 140, 131: 141, 132: 142, 133: 143, 134: 144, 135: 145, 136: 146, 137: 147, 138: 148, 139: 149, 140: 150, 141: 151, 142: 152, 143: 153, 144: 154, 145: 155, 146: 156, 147: 157, 148: 158, 149: 159, 150: 160, 151: 161, 152: 162, 153: 163, 154: 164, 155: 165, 156: 166, 157: 167, 158: 168, 159: 169, 160: 170, 161: 171, 162: 172, 163: 173, 164: 174, 165: 175, 166: 176},
    'Codec':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18},
    'Compress':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47},
    'Gson':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18},
    'Cli':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 16, 14: 17, 15: 18, 16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40},
    'Jsoup':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 17, 16: 19, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 56, 54: 57, 55: 58, 56: 59, 57: 60, 58: 61, 59: 62, 60: 63, 61: 64, 62: 65, 63: 66, 64: 67, 65: 68, 66: 70, 67: 71, 68: 72, 69: 73, 70: 74, 71: 75, 72: 76, 73: 77, 74: 78, 75: 79, 76: 80, 77: 81, 78: 82, 79: 83, 80: 84, 81: 85, 82: 86, 83: 87, 84: 88, 85: 89, 86: 90, 87: 91, 88: 92, 89: 93},
    'Csv':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16},
    'JacksonCore': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20},
    'JacksonXml': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    'Collections': {0: 25, 1: 26}
}

def splitCamel(token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
p = pickle.load(open(pr + 'res_%d_%s_%s.pkl'%(seed,lr,batch_size), 'rb'))
f = pickle.load(open(pr + '.pkl', 'rb'))

# print(len(f), len(p))
#assert(0)
score = []
score2 = []
eps = {}
best_ids = []
for _, i in enumerate(p):
    maxn = 1e9
    xs = p[i]
    score.extend(xs[0])
    # print(xs)
    # print(i, xs[0], xs[1], xs[3])
    # print('###############')
    # print(i)
    # print('--------------')
    # print(xs[0])
    # print('--------------')
    # print(xs[1])
    # print('--------------')
    # print(xs[3])
    # print('###############')
    minl = 1e9
    for x in f[i]['ans']:
        m = xs[1].index(x)
        minl = min(minl, m)
    score2.append(minl)
    rrdic = {}
    # for x in f[i]['methods']:
    #     rrdic[f[i]['methods'][x]] = x#".".join(x.split(":")[0].split(".")[-2:])
    #rrdict = {}
    #for s in f[i]['ftest']:
    #    rrdict[f[i]['ftest'][s]] = ".".join(s.split(":")[0].split(".")[-2:])
    # for x in f[i]['ftest']:
    #     print(splitCamel(".".join(x.split(":")[0].split(".")[-2:])), x, ".".join(x.split(":")[0].split(".")[-2:]))
    # print("-----")
    # for x in f[i]['ans']:
    #     print(splitCamel(rrdic[x]), rrdic[x], ',')
    # print("-----")
    # print(rrdic, f[i]['ans'])
    # print(splitCamel(rrdic[xs[1][0]]), rrdic[xs[1][0]], ',', xs[1][0], f[i]['ans'])
    #print(f[i]['methods'], f[i]['ftest'], f[i]['ans'])
    for x in xs[2]:
        if x in eps:
            eps[x] += 1
        else:
            eps[x] = 1
    if 10 in xs[2]:
        best_ids.append(i)
    #print(xs[2])
    #score.append(maxn)

# print(score)

with open(pr + 'result_final_%d_%s_%s'%(seed,lr, batch_size), 'w') as pp:
    pp.write("lr: %f seed %d batch_size %d\n"%(lr, seed, batch_size))
    pp.write('num: %s\n'%len(p))
    # pp.write('%d: %d\n'%(10, eps[10]))
    pp.write(str(sorted(eps.items(), key=lambda x:x[1])))

# print(len(score))
a = []
for i, x in enumerate(score):
    if x != 0:
        a.append(i)
# print(a)
# print(len(score))
# print(score.count(0))
# print(score2.count(0))
# print(eps)
c1 = 0
for x in score:
    if x < 3:
        c1 += 1
c2 = 0
for x in score:
    if x < 5:
        c2 += 1
# print('top35',c1, c2)
# print(sorted(eps.items(), key=lambda x:x[1]))

# print(best_ids)
# print(len(best_ids))


best_epoch = sorted(eps.items(), key=lambda x:x[1])[-1][0]
top1 = 0
top3 = 0
top5 = 0
top10 = 0
mfr = []
mar = []
for idx in p:
    xs = p[idx]
    each_epoch_pred = xs[3]
    best_pred = each_epoch_pred[best_epoch]
    score_pred = each_epoch_pred[str(best_epoch)+'_pred']
    # print('-'*20)
    print('Project Number:', dmap[pr][idx])
    print('Correct Answer:', f[idx]['ans'])
    print(best_pred)
    print(score_pred)
    ar = []
    minl = 1e9
    to1 = 0
    to3 = 0
    to5 = 0
    to10 = 0
    for x in f[idx]['ans']:
        m = best_pred.index(x)
        ar.append(m)
        minl = min(minl, m)
    if minl == 0:
        top1 += 1
        to1 = 1
    if minl < 3:
        top3 += 1
        to3 = 1
    if minl < 5:
        top5 += 1
        to5 = 1
    if minl < 10:
        top10 += 1
        to10 = 1
    mfr.append(minl)
    mar.append(np.mean(ar))
    print('Top1:', to1)
    print('Top3:', to3)
    print('Top5:', to5)
    print('Top10:', to10)
    # print('-'*20)
result_path = os.path.join("result-all")
if not os.path.exists(result_path):
    os.makedirs(result_path)

# print('########Original#########')
# print('top1:',top1)
# print('top3:',top3)
# print('top5:',top5)
# print('top10:',top10)
# print('mfr:',np.mean(mfr))
# print('mar:',np.mean(mar))
# print('###############')

# with open(result_path + '/' + pr, 'w') as f:
#     f.write('top1: %d\n'%top1)
#     f.write('top3: %d\n'%top3)
#     f.write('top5: %d\n'%top5)
#     f.write('mfr: %f\n'%np.mean(mfr))
#     f.write('mar: %f\n'%np.mean(mar))
# best_epoch = sorted(eps.items(), key=lambda x:x[1])[-1][0]
top_count = [0] * 5  # list to count correct items in each position
mfr = []
mar = []
for idx in p:
    xs = p[idx]
    each_epoch_pred = xs[3]
    best_pred = each_epoch_pred[best_epoch]
    score_pred = each_epoch_pred[str(best_epoch)+'_pred']
    # print('-'*20)
    # print('Project Number:', idx)
    # print('Correct Answer:', f[idx]['ans'])
    # print('Ranking positions:',best_pred)
    # print('Scores:',score_pred)
    ar = []
    minl = 1e9
    for x in f[idx]['ans']:
        m = best_pred.index(x)
        ar.append(m)
        minl = min(minl, m)
        if m < len(top_count):  # increment the count if the index is within the list length
            top_count[m] += 1
    mfr.append(minl)
    mar.append(np.mean(ar))

    # calculate top-k values
    top1 = top_count[0]
    top3 = sum(top_count[:3])
    top5 = sum(top_count)

    # print('Current Top1:', top1)
    # print('Current Top3:', top3)
    # print('Current Top5:', top5)
    # print('-'*20)

# result_path = os.path.join("result-all")
# if not os.path.exists(result_path):
#     os.makedirs(result_path)

# print('########GBMFL########')
# print('Final top1:',top1)
# print('Final top3:',top3)
# print('Final top5:',top5)
# print('mfr:',np.mean(mfr))
# print('mar:',np.mean(mar))
# print('################')

# with open(result_path + '/' + pr, 'w') as f:
#     f.write('top1: %d\n' % top1)
#     f.write('top3: %d\n' % top3)
#     f.write('top5: %d\n' % top5)
#     f.write('mfr: %f\n' % np.mean(mfr))
#     f.write('mar: %f\n' % np.mean(mar))
