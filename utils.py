import argparse


class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.00002, help="Learning rate")
        parser.add_argument('--epochPRE', type=int, default=40, help="Number of epoch on pretraining")
        parser.add_argument('--epochRL', type=int, default=15, help="Number of epoch on training with RL")
        parser.add_argument('--dim', type=int, default=300, help="Dimension of embeddings")
        parser.add_argument('--statedim', type=int, default=300, help="Dimension of state")
        parser.add_argument('--batchsize', type=int, default=4, help="Batch size on training")
        parser.add_argument('--batchsize_test', type=int, default=16, help="Batch size on testing")
        parser.add_argument('--print_per_batch', type=int, default=50, help="Print results every XXX batches")
        parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")
        parser.add_argument('--numprocess', type=int, default=1, help="Number of process")
        parser.add_argument('--start', type=str, default='', help="Directory to load model")
        parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
        parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
        parser.add_argument('--datapath', type=str, default='./data/14lap/', help="Data directory")
        parser.add_argument('--testfile', type=str, default='test_triplets.txt', help="Filename of test file")
        parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")
        parser.add_argument('--seed', type=int, default=1, help="PyTorch seed value")
        return parser


def calcF1(acc, cnt, tot, beta=1.0):
    """
    Get F1 score.
    """
    if cnt == 0 or tot == 0:
        return 0
    precision = float(acc) / float(cnt)
    recall = float(acc) / float(tot)
    if precision + recall < 1e-5:
        return 0
    return (1+beta*beta) * precision * recall / (beta*beta*precision + recall)


def calc_acc(top_action, bot_opinion_action, gold_labels, label_triple, mode):
    #TopActions, BotOpinionActions, BotSentimentActions = gold_labels
    if 'NER' not in mode:
        label_taget_set = set()
        for label in label_triple:
            label_taget_set.add(str(label[0]))
        pred_target_set = set()
        for i in range(len(top_action)):
            if top_action[i] >= 2:
                s_pos = find_start(top_action, i, top_action[i]-1)
                target = [t for t in range(s_pos, i+1)]
                pred_target_set.add(str(target))
        acc, cnt, tot = 0, len(pred_target_set), len(label_taget_set)
        for label in label_taget_set:
            for pred in pred_target_set:
                if label == pred:
                    acc += 1
        return acc, tot, cnt
    
    if 'RE' not in mode:
        label_taget_set = set()
        for label in label_triple:
            label_taget_set.add(str(label[1]))
        pred_target_set = set()
        for i in range(len(top_action)):
            if top_action[i] > 3:
                s_pos = find_start(top_action, i, top_action[i]-3)
                target = [t for t in range(s_pos, i+1)]
                pred_target_set.add(str(target))
        acc, cnt, tot = 0, len(pred_target_set), len(label_taget_set)
        for label in label_taget_set:
            for pred in pred_target_set:
                if label == pred:
                    acc += 1
        return acc, tot, cnt
    
    pred_triple = []
    j = 0
    #print('top_action', top_action)
    for i in range(len(top_action)):
        if top_action[i] > 1:
            s_pos = find_start(top_action, i, top_action[i]-1)
            target = [t for t in range(s_pos, i+1)]
            #print('bot_action', bot_action)
            #print(bot_action[j])
            for k in range(len(bot_opinion_action[j])):
                if bot_opinion_action[j][k] > 3:
                    s_pos = find_start(bot_opinion_action[j], k, bot_opinion_action[j][k]-3)
                    opinion = [t for t in range(s_pos, k+1)]
                    #one_sen = []
                    #one_sen.append(target)
                    #one_sen.append(opinion)
                    if bot_opinion_action[j][k] == 4:
                        sentiment = 'POS'
                        #one_sen.append('POS')
                    if bot_opinion_action[j][k] == 5:
                        sentiment = 'NEG'
                        #one_sen.append('NEG')
                    if bot_opinion_action[j][k] == 6:
                        sentiment = 'NEU'
                        #one_sen.append('NEU')
                    
                    #pred_triple.append(one_sen)
                    pred_triple.append((target, opinion, sentiment))
            j += 1
    print(gold_labels[0], '  ', label_triple)
    print(top_action, '  ', pred_triple)
    print('\n')
    acc, cnt, tot = 0, len(pred_triple), len(label_triple)
    for label in label_triple:
        for pred in pred_triple:
            #print("label:", label)
            #print("pred:", pred)
            if label == pred:
                acc += 1
    return acc, tot, cnt


def rule_actions(gold_labels):
    return gold_labels[0], gold_labels[1]
				
def find_start(tags, i, num):
    for j in range(i)[::-1]:
        if tags[j] != num:
            return j+1
    return 0

def find_end(tags, i, num):
    for j in range(i+1, len(tags)):
        if tags[j] != num:
            return j
    return len(tags)



