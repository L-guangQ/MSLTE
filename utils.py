import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from datetime import datetime

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        # params = torch.tensor([1.,1.,2.], requires_grad=True)
        self.params = torch.nn.Parameter(params)
        # print(self.params)
        self.eps = 1e-8

    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        # self.params = F.relu(self.params)
        for i, loss in enumerate(x):
            loss_sum += 1 / (self.params[i] ** 2 + self.eps) * loss #+ torch.log(torch.abs(self.params[i]))

            # if i == length:
            #     loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # else:
            #     loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def cacl_acc(args):
    dataset = args.dataset
    denominator = 45
    part2of3 = 0.0
    part1of3 = 0.0
    total = []
    tmp = []
    top2_arr = []
    top1_arr = []
    f = open(f'./result/{dataset}_meanACC.txt', 'a+')
    for sub in range(15):
        for session in range(3):
            ck = torch.load(f'./save_model/{dataset}_supervised_jointly_train/checkpoint_s{str(sub+1).zfill(2)}_{session+1}.pkl', map_location='cuda:0')
            print(ck['acc'])
            acc = ck['acc']
            # f.write(f's{str(sub+1).zfill(2)}_{session+1}\t{acc:.4f}\n')
            total.append(ck['acc'])
            tmp.append(ck['acc'])
            # arr.append(ck['ACC'])

        tmp.sort(reverse=False)
        top2_arr.append(tmp[1])
        top2_arr.append(tmp[2])
        part2of3 = part2of3 + tmp[1] + tmp[2]
        top1_arr.append(tmp[2])
        tmp = []
    print('total: {:.4f}   std: {:.4f}'.format(np.mean(total), np.std(total)))     # for SEED subject-dependent
    print('2/3: {:.4f}   std: {:.4f}'.format(np.mean(top2_arr), np.std(top2_arr)))
    print('1/3: {:.4f}   std: {:.4f}'.format(np.mean(top1_arr), np.std(top1_arr))) # for SEED subject-independent
    f.write(f'\n---------- ' + str(datetime.now()) + ' ----------')
    f.write(f'\n k = {args.cheb_k}  mask_rate={args.mask_rate}')
    f.write(f'\n{dataset}:   total_mean_acc:{np.mean(total):.4f} std: {np.std(total):.4f}    2/3_mean_acc:{np.mean(top2_arr):.4f} std:{np.std(top2_arr):.5f}'
            f'   1/3_mean_acc:{np.mean(top1_arr):.4f} std:{np.std(top1_arr):.5f}\n')
    f.close()

