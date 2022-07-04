import time
import torch.optim as optim
import queue
from utils import calc_acc, rule_actions
from optimize import optimize_round


def workProcess(model, datas, sample_round, mode, device, sentiments, test):
    """
    Get model outputs and train model.
    """
    acc, cnt, tot = 0, 0, 0
    loss = .0
    for data in datas:
        top_actions, top_actprobs, bot_opinion_actions, bot_opinion_actprobs = [], [], [], []
        if not test:
            preoptions, pre_opinion_actions = rule_actions(data['triplets'])
        bert_to_whitespace = data['bert_to_whitespace']
        whitespace_tokens = data['whitespace_tokens']
        for i in range(sample_round):
            # pretraining
            if "pretrain" in mode and "test" not in mode:
                top_action, top_actprob, bot_opinion_action, bot_opinion_actprob = \
                        model(mode, data['pos_tags'], data['sentext'], \
                        preoptions, pre_opinion_actions, device)
            # train from scratch
            else:
                top_action, top_actprob, bot_opinion_action, bot_opinion_actprob = \
                        model(mode, data['pos_tags'], data['sentext'], None, None, device)

            top_actions.append(top_action)
            top_actprobs.append(top_actprob)
            bot_opinion_actions.append(bot_opinion_action)
            bot_opinion_actprobs.append(bot_opinion_actprob)
            

            print('sentence:', data['sentext'])
            acc1, tot1, cnt1 = calc_acc(top_action, bot_opinion_action, \
                    data['triplets'], data['labels'], mode)
            acc += acc1
            tot += tot1
            cnt += cnt1
            
        # training optimisation
        if "test" not in mode:
            loss += optimize_round(model, top_actions, top_actprobs, bot_opinion_actions,\
                    bot_opinion_actprobs, data['triplets'], data['labels'], mode, device)
        
    if len(datas) == 0:
        return 0, 0, 0, 0
    
    return acc, cnt, tot, loss / len(datas)


def worker(model, rank, dataQueue, resultQueue, freeProcess, lock, flock, lr, sentiments):
    # get data from queue to train/val/test
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Process ", rank, " start service.")
    flock.acquire()
    freeProcess.value += 1
    flock.release()
    while True:
        datas, sample_round, mode, dataID, device, sentiments, test = dataQueue.get()
        flock.acquire()
        freeProcess.value -= 1
        flock.release()
        model.zero_grad()
        acc, cnt, tot, loss = workProcess(model, datas, sample_round, mode, device, sentiments, test)
        resultQueue.put((acc, cnt, tot, dataID, rank, loss))
        if not "test" in mode:
            lock.acquire()
            optimizer.step()
            lock.release()
        flock.acquire()
        freeProcess.value += 1
        flock.release()


def train(dataID, model, datas, sample_round, mode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test):
    # put data into queue
    dataPerProcess = len(datas) // numprocess
    while freeProcess.value != numprocess:
        pass
    acc, cnt, tot = 0, 0, 0
    loss = .0
    for r in range(numprocess):
        endPos = ((r+1)*dataPerProcess if r+1 != numprocess else len(datas))
        data = datas[r*dataPerProcess: endPos]
        dataQueue.put((data, sample_round, mode, dataID, device, sentiments, test))
    lock.acquire()
    try:
        for r in range(numprocess):
            while True:
                item = resultQueue.get()
                if item[3] == dataID:
                    break
                else:
                    print ("receive wrong dataID: ", item[3], "from process ", item[4])
            acc += item[0]
            cnt += item[1]
            tot += item[2]
            loss += item[5]
    except queue.Empty:
        print("The result of some process missed...")
        print(freeProcess.value)
        lock.release()
        time.sleep(2)
        print(freeProcess.value)
        while True:
            pass

    lock.release()
    return (acc, cnt, tot)


def test(dataID, model, datas, mode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test):
    testmode = mode + ["test"]
    if dataID < -2:
        print(testmode)
    return train(-dataID-1, model, datas, 1, testmode, dataQueue, resultQueue, freeProcess, lock, numprocess, device, sentiments, test)

