import numpy as np
import torch
from utils import calc_acc, calcF1


def calcTopTagReward(top_action, gold_labels):
    """
    Intermediate top reward.
    """
    gold = gold_labels[0]
    lenth = len(top_action)
    r = [0. for i in range(lenth)]
    for i in range(lenth)[::-1]:
        if gold[i] == top_action[i]:
            if top_action[i] == 2:
                r[i] = 0.5
            elif top_action[i] == 1:
                r[i] = 0.2
        else:
            r[i] = -0.5
    return r

def calcTopMulReward(top_action, gold_labels):
    gold = gold_labels[0]
    lenth = len(top_action)
    r = [0. for i in range(lenth)]
    for i in range(lenth)[::-1]:
        # multi tag based reward
        if top_action[i] > 1:
            flag = False
            if gold[i] == top_action[i]:
                flag = True
                s_pos = find_start(gold, i, top_action[i]-1)
                for t in range(s_pos, i + 1):
                    if top_action[t] != gold[t]:
                        flag = False
            if flag:
                r[i] = 0.5
            else:
                r[i] = -0.5
    return r


def calcTopF1Reward(top_action, gold_labels, label_triple, top_bias = 0.):
    """
    F1 top reward, using F1 score.
    """
    r = 0.
    a1, t1, c1 = calc_acc(top_action, None, gold_labels, label_triple, ["RE"])
    if c1 != 0:
        r = calcF1(a1, c1, t1)
    else:
        r = -2
    if c1 > t1:
        r -= 0.5 * (c1 - t1)
    r *= len(top_action)
    return r - top_bias


def calcBotTagReward(top_action, bot_opinion_action, gold_labels, mode):
    """
    Intermediate bottom reward.
    """
    TopActions, BotOpinionActions = gold_labels
    lenth = len(top_action)
    r = [[0. for i in range(lenth)] for j in range(len(bot_opinion_action))]
    j = 0
    for i in range(lenth):
        # tag-based reward
        if top_action[i] > 1:
            if TopActions[i] == top_action[i]:
                flag = True
                print('dataset:', mode[0])
                if mode[0] != '14lap' and mode[0] != '16res':
                    s_pos = find_start(TopActions, i, top_action[i]-3)
                    for t in range(s_pos, i + 1):
                        if top_action[t] != TopActions[t]:
                            flag = False
                opi = BotOpinionActions[i]
                if flag:
                    for t in range(lenth):
                        if opi[t] == bot_opinion_action[j][t]:
                            if opi[t] > 3:
                                r[j][t] = 0.5
                            elif opi[t] > 0:
                                r[j][t] = 0.2
                        else:
                            r[j][t] = -0.5
            j += 1
    return r


def calcBotMulReward(top_action, bot_opinion_action, gold_labels, mode):
    TopActions, BotOpinionActions = gold_labels
    lenth = len(top_action)
    r = [[0. for i in range(lenth)] for j in range(len(bot_opinion_action))]
    j = 0
    for i in range(lenth):
        # multi tag based reward
        if top_action[i] > 1:
            if TopActions[i] == top_action[i]:
                flag = True
                
                if mode[0] != '14lap' and mode[0] != '16res':
                    s_pos = find_start(TopActions, i, top_action[i]-3)
                    for t in range(s_pos, i + 1):
                        if top_action[t] != TopActions[t]:
                            flag = False
                opi = BotOpinionActions[i]
                if flag:
                    for t in range(lenth):
                        # multi tag based reward
                        if bot_opinion_action[j][t] > 3:
                            b_flag = False
                            if opi[t] == bot_opinion_action[j][t]:
                                s_pos = find_start(opi, t, bot_opinion_action[j][t]-3)
                                b_flag = True
                                for s in range(s_pos, t+1):
                                    if bot_opinion_action[j][s] != opi[s]:
                                        b_flag = False
                            if b_flag:
                                r[j][t] = 0.5
                            else:
                                r[j][t] = -0.5
            j += 1
    return r


def calcBotF1Reward(top_action, bot_opinion_action, gold_labels, label_triple, bot_bias = 0.):
    """
    F1 Bot reward, using F1 score.
    """
    TopActions, BotOpinionActions = gold_labels
    lenth = len(top_action)
    r = [0. for j in range(len(bot_opinion_action))]
    j = 0
    for i in range(lenth):
        if top_action[i] > 1:
            if TopActions[i] == top_action[i]:
                opi = BotOpinionActions[i]
                a1, t1, c1 = calc_acc(bot_opinion_action[j], None, gold_labels, label_triple, ["NER"])
                if c1 != 0:
                    r[j] = calcF1(a1, c1, t1)
                else:
                    r[j] = -2
                if c1 > t1:
                    r[j] -= 0.5 * (c1 - t1)
                r[j] *= len(top_action)
            j += 1
    for j in range(len(bot_opinion_action)):
        r[j] -= bot_bias
    return r


def calcTopGrad(top_action, top_actprob, top_tag_reward, top_mul_reward, top_f1_reward, pretrain=False, device=torch.device("cpu")):
    lamda = 0
    lenth = len(top_action)
    grads = torch.FloatTensor(1, ).fill_(0).to(device)
    top_final_reward = sum(top_mul_reward) / lenth + top_f1_reward
    for i in range(lenth)[::-1]:
        top_tot_reward = top_f1_reward + top_tag_reward[i]
        to_grad = -torch.log(top_actprob[i]).to(device)
        if not pretrain:
            to_grad *= torch.FloatTensor(1, ).fill_(top_tot_reward).to(device)
        '''
        if top_action[i] == 0:
            to_grad *= 0.3
        elif top_action[i] == 1:
            to_grad *= 0.7
        '''
        grads = grads + to_grad
    return grads


def calcBotGrad(top_action, bot_opinion_action, bot_opinion_actprob, bot_tag_reward, bot_mul_reward, bot_f1_reward, pretrain=False, device=torch.device("cpu")):
    lamda = 0
    lenth = len(top_action)
    bot_final_reward = [0. for i in range(lenth)]
    grads = torch.FloatTensor(1, ).fill_(0).to(device)
    grads = torch.unsqueeze(grads, dim=0)
    j = 0
    for i in range(lenth):
        if top_action[i] > 1:
            bot_final_reward[i] = sum(bot_mul_reward[j]) / lenth + bot_f1_reward[j]
            for k in range(lenth)[::-1]:
                bot_final_reward[i] = bot_final_reward[i] + bot_tag_reward[j][k]
                opinion_to_grad = -torch.log(bot_opinion_actprob[j][k])
                opinion_to_grad = torch.unsqueeze(opinion_to_grad, dim=0).to(device)
                if not pretrain:
                    opinion_to_grad *= torch.FloatTensor(1, ).fill_(bot_final_reward[i]).to(device)
                '''
                if bot_opinion_action[j][k] == 0:
                    opinion_to_grad *= 0.3
                elif bot_opinion_action[j][k] <= 3:
                    opinion_to_grad *= 0.7
                else:
                    opinion_to_grad *= 1.0
                '''
                grads = grads + opinion_to_grad
            j += 1
    return bot_final_reward, grads


def optimize(model, top_action, top_actprob, bot_opinion_action, bot_opinion_actprob, gold_labels, label_triple, mode, top_bias = 0., bot_bias = 0., device=torch.device("cpu")):
    lenth = len(top_action)
    top_tag_reward = calcTopTagReward(top_action, gold_labels)
    top_mul_reward = calcTopMulReward(top_action, gold_labels)
    top_f1_reward = calcTopF1Reward(top_action, gold_labels, label_triple, top_bias)
    pretrain = True if "pretrain" in mode else False
    if "NER" in mode:
        bot_reward = calcBotTagReward(top_action, bot_opinion_action, gold_labels, mode)
        bot_mul_reward = calcBotMulReward(top_action, bot_opinion_action, gold_labels, mode)
        bot_f1_reward = calcBotF1Reward(top_action, bot_opinion_action, gold_labels, label_triple, bot_bias)
        bot_tot_reward, bot_grads = calcBotGrad(top_action, bot_opinion_action, bot_opinion_actprob, bot_reward, bot_mul_reward, bot_f1_reward, pretrain, device)
        for i in range(lenth):
            top_tag_reward[i] += bot_tot_reward[i]
    else:
        bot_grads = torch.FloatTensor(1, ).fill_(0).to(device)
        bot_grads = torch.unsqueeze(grads, dim=0)
    if "RE" in mode:
        top_grads = calcTopGrad(top_action, top_actprob, top_tag_reward, top_mul_reward, top_f1_reward, pretrain, device)
    grads = top_grads + bot_grads
    loss = grads.cpu().data[0]
    grads.backward()
    return loss


def optimize_round(model, top_actions, top_actprobs, bot_opinion_actions, bot_opinion_actprobs, gold_labels, label_triple, mode, device):
    sample_round = len(top_actions)
    # get bias first
    if "RE" in mode:
        top_bias = 0.
        for i in range(sample_round):
            top_bias += calcTopF1Reward(top_actions[i], gold_labels, label_triple, 0.)
        top_bias /= sample_round
    else:
        top_bias = 0.
    if "NER" in mode:
        bot_bias, bot_cnt = 0., 0
        for i in range(sample_round):
            tmp = calcBotF1Reward(top_actions[i], bot_opinion_actions[i], gold_labels, label_triple, 0.)
            bot_cnt += len(tmp)
            bot_bias += np.sum(tmp)
        if bot_cnt != 0:
            bot_bias /= bot_cnt
    else:
        bot_bias = 0.
    loss = .0
    # real optimisation with top/bot biases taken into account
    for i in range(sample_round):
        loss += optimize(model, top_actions[i], top_actprobs[i], bot_opinion_actions[i], \
                bot_opinion_actprobs[i], gold_labels, label_triple, mode, top_bias, bot_bias, device)
    return loss / sample_round

def find_start(tags, i, num):
    for j in range(i)[::-1]:
        if tags[j] != num:
            return j+1
    return i

def find_end(tags, i, num):
    for j in range(i+1, len(tags)):
        if tags[j] != num:
            return j
    return len(tags)
