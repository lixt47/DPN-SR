import random, sys, time, os
import torch
from data_reader import DataReader
from dpn import Model
from utils import Parser
from trainer import train, test, worker
import torch.multiprocessing as mp
from utils import calcF1
import datetime
import json
import numpy as np
#torch.multiprocessing.set_start_method('spawn')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time

def work(mode, train_data, test_data, dev_data, model, args, sampleround, epoch, device, experiment_id):
    top_dev_f1 = 0
    best_epoch = 0
    log_f = open("checkpoints/" + str(experiment_id) + "/log.txt", 'a')
    log_f.write(args.datapath + '\n')
    log_f.close()
    
    for e in range(epoch):
        random.shuffle(train_data)
        print("Train epoch", e, "...")

        # training
        batchcnt = (len(train_data) - 1) // args.batchsize + 1
        for b in range(batchcnt):
            start = time.time()
            data = train_data[b * args.batchsize : (b+1) * args.batchsize]
            acc, cnt, tot = train(b, model, data, sampleround, \
                    mode, dataQueue, resultQueue, freeProcess, lock, args.numprocess, device, sentiments, args.test)
            trainF1 = calcF1(acc, cnt, tot)
            # print time per batch
            if b % args.print_per_batch == 0:
                print("Train batch", b, ": F1=", trainF1, ", time=", (time.time() - start))

        with torch.no_grad():
            # validation
            batchcnt = (len(dev_data) - 1) // args.batchsize_test + 1
            acc, cnt, tot = 0, 0, 0
            for b in range(batchcnt):
                data = dev_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
                acc_, cnt_, tot_ = test(b, model, data, mode, \
                        dataQueue, resultQueue, freeProcess, lock, args.numprocess, device, sentiments, args.test)
                acc += acc_
                cnt += cnt_ 
                tot += tot_
            devF1 = calcF1(acc, cnt, tot)

            # testing
            batchcnt = (len(test_data) - 1) // args.batchsize_test + 1
            acc, cnt, tot = 0, 0, 0
            for b in range(batchcnt):
                data = test_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
                acc_, cnt_, tot_ = test(b, model, data, mode, \
                        dataQueue, resultQueue, freeProcess, lock, args.numprocess, device, sentiments, args.test)
                acc += acc_
                cnt += cnt_ 
                tot += tot_
            testF1 = calcF1(acc, cnt, tot)
            if cnt == 0 or tot == 0:
                testP = 0
                testR = 0
            else:
                testP = float(acc) / float(cnt)
                testR = float(acc) / float(tot)

            # save stats and model
            print("Epoch ", e, ": dev F1=", devF1, ", test F1=", testF1)
            print("test F1=", testF1, "test R=", testR, "test P=", testP)
            log_f = open("checkpoints/" + str(experiment_id) + "/log.txt", 'a')
            log_f.write("Epoch " + str(e) + ": dev F1=" + str(devF1) + ", test F1=" + str(testF1) + "\n")
            log_f.write("test F1=" + str(testF1) + "test R=" + str(testR) + "test P=" + str(testP) + "\n")
            #log_f.close()
            if devF1 > top_dev_f1:
                best_epoch = e
                torch.save(model, "checkpoints/" + str(experiment_id) + "/model")
                best_results = "Epoch " + str(e) + ": dev F1=" + str(devF1) + ", test F1=" + str(testF1)
                best_f = open("checkpoints/" + str(experiment_id) + "/best.txt", 'w')
                best_f.write(args.datapath + '\n')
                best_f.write(best_results)
                best_f.close()
            top_dev_f1 = max(devF1, top_dev_f1)
    print('best_epoch:', best_epoch)
    log_f.write('best_epoch:' + str(best_epoch) + "\n")
    log_f.close()


if __name__ == "__main__":
    # get args
    print('Current Time:', time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    argv = sys.argv[1:]
    parser = Parser().getParser()
    args, _ = parser.parse_known_args(argv)

    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading data...")
    dm = DataReader(args.datapath, args.testfile, args.test)
    all_pos_tags = dm.all_pos_tags
    sentiments = dm.sentiments
    print('Dataset:', dm.dataset)

    if not args.test:
        train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
        print("#train_data:", len(train_data))
        print("#dev_data:", len(dev_data))
        print("#test_data:", len(test_data))
    else:
        test_data = dm.data['test']
        print("#test_data:", len(test_data))

    if not args.test:
        if not os.path.exists('checkpoints'):
             os.mkdir('checkpoints')
        if not os.path.exists('checkpoints/{}'.format(experiment_id)):
            os.mkdir('checkpoints/{}'.format(experiment_id))
        with open('checkpoints/{}/args.json'.format(experiment_id), 'w') as f:
            json.dump(vars(args), f)
            f.close()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(args.lr, args.dim, args.statedim, dm.sent_count, args.dropout, all_pos_tags)
    model.to(device)
    if args.start != '':
        # if pretrained model exists
        pretrain_model = torch.load(args.start, map_location='cpu') 
        model_dict = model.state_dict()
        pretrained_dict = pretrain_model.state_dict() 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict) 
    model.share_memory()
    try:
        #mp.set_start_method('spawn')
        ctx = mp.get_context("spawn")
    except RuntimeError:
        pass
    
    # start PyTorch multiprocessing
    processes = []
    #dataQueue = mp.Queue()
    dataQueue = ctx.Queue()
    #resultQueue = mp.Queue()
    resultQueue = ctx.Queue()
    #freeProcess = mp.Manager().Value("freeProcess", 0)
    freeProcess = ctx.Manager().Value("freeProcess", 0)
    #lock = mp.Lock()
    lock = ctx.Lock()
    #flock = mp.Lock()
    flock = ctx.Lock()
    print("Starting training service, overall process number: ", args.numprocess)
    for r in range(args.numprocess):
        #p = mp.Process(target=worker, args= \
        #        (model, r, dataQueue, resultQueue, freeProcess, lock, flock, args.lr, sentiments))
        p = ctx.Process(target=worker, args= \
                (model, r, dataQueue, resultQueue, freeProcess, lock, flock, args.lr, sentiments))
        p.start()
        processes.append(p)
    
    # start work
    
    if args.test:
        batchcnt = (len(test_data) - 1) // args.batchsize_test + 1
        acc, cnt, tot = 0, 0, 0
        for b in range(batchcnt):
            data = test_data[b * args.batchsize_test : (b+1) * args.batchsize_test]
            acc_, cnt_, tot_ = test(b, model, data, [dm.dataset, "RE", "NER"], \
                    dataQueue, resultQueue, freeProcess, lock, args.numprocess, device, sentiments, args.test)
            acc += acc_
            cnt += cnt_ 
            tot += tot_
            # print(acc, cnt, tot)
        testF1 = calcF1(acc, cnt, tot)
        print("test P: ", acc/cnt, "test R: ", acc/tot, "test F1: ", testF1)
    elif args.pretrain:
        work([dm.dataset, "RE", "NER", "pretrain"], train_data, test_data, dev_data, model, args, 1, args.epochPRE, device, experiment_id)
    else:
        work([dm.dataset, "RE", "NER"], train_data, test_data, dev_data, model, args, args.sampleround, args.epochRL, device, experiment_id)

    for p in processes:
        p.terminate()
