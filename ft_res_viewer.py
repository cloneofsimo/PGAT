import torch

jt = torch.load("metric_jt.res")
pt = torch.load("metric_pt.res")

print(jt, pt)


PA_set = [1, 2, 3, 4]
PA_space = {}

met = [[0 for _ in range(4)] for _ in range(4)]

for Pa in PA_set:
    with open(f"PA_{Pa}.txt", 'r') as F_tr:
        whole = F_tr.read()
        whole = whole.split('\n')

        for word in whole:
            if len(word) > 10:
                print(word)
        whole = set(whole)

        print(f"{Pa} has {len(whole)} elements")
        PA_space[Pa] = whole


for Pa in PA_set:
    for Pb in PA_set:
        if Pa == Pb:
            continue
        coll = 0
        for ele in PA_space[Pa]:
            if ele in PA_space[Pb]:
                coll += 1
        
        met[Pa-1][Pb-1] = coll

print(met)
            