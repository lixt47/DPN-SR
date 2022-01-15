import json
import ast
import spacy
from spacy.tokens import Doc
from transformers import BertTokenizer
import tokenizations
import os


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # all tokens have a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class DataManager:
    def __init__(self, path, testfile, test):
        # POS tagger
        nlp = spacy.load("en_core_web_sm")
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        # BERT tokeniser
        bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

        # process data
        # NOTE: order for self.all_pos_tags depends on dataset, order has to be the same between dataset splits
        self.all_pos_tags = []
        self.data = {}
        if not test:
            names = ["train", "dev", "test"]
        else:
            names = ["test"]
        for name in names:
            self.data[name] = []
            filename = testfile if name == "test" else name + "_triplets.txt"
            with open(os.path.join(path, filename)) as fl:
                for line in fl.readlines():
                    # process ASTE data for HRL
                    sentence, triplets = line.strip().split('####')
                    # tokenize for BERT
                    whitespace_tokens = sentence.split()
                    bert_tokens = bert_tokenizer.tokenize(sentence)
                    # https://github.com/tamuhey/tokenizations
                    whitespace_to_bert, bert_to_whitespace = tokenizations.get_alignments(whitespace_tokens, bert_tokens)
                    # get WordPiece POS tags
                    doc = nlp(sentence)
                    #print('doc:', doc)
                    pos_tags = ([w.pos_ for w in doc])
                    #print('pos_tags:', pos_tags)
                    for pt in pos_tags:
                        pt_b = 'B-' + pt
                        pt_i = 'I-' + pt
                        if pt_b not in self.all_pos_tags:
                            self.all_pos_tags.append(pt_b)
                        if pt_i not in self.all_pos_tags:
                            self.all_pos_tags.append(pt_i)
                    bert_pos_tags = []
                    for i, pos_tag in enumerate(pos_tags):
                        if len(whitespace_to_bert[i]) > 1:
                            bert_pos_tags.append('B-' + pos_tag)
                            for j in range(len(whitespace_to_bert[i])-1):
                                bert_pos_tags.append('I-' + pos_tag)
                        else:
                            bert_pos_tags.append('B-' + pos_tag)

                    test = False
                    if not test:
                        r_set = {}
                        triplets = ast.literal_eval(triplets)
                        all_triplets = []
                        gold_labels = []
                        triplet_pos = []
                        for i, triplet in enumerate(triplets):
                            aspect_ids, opinion_ids, sentiment = triplet
                            aspect, opinion = [], []
                            if str(aspect_ids) not in r_set.keys():
                                final_triplet = {}
                                #final_triplet['sentpol'] = [0 for i in range(len(bert_tokens))]
                                final_triplet['aspect'] = ''
                                final_triplet['opinion'] = ''
                                final_triplet['aspect_tags'] = [0 for i in range(len(bert_tokens))]
                                final_triplet['opinion_tags'] = [0 for i in range(len(bert_tokens))]
                                r_set[str(aspect_ids)] = final_triplet
                            
                            final_triplet = r_set[str(aspect_ids)]
                            # align tokens between ASTE-Data-V2's whitespace tokenisation and BERT tokenisation
                            for j, aspect_id in enumerate(aspect_ids):
                                bert_aspect_ids = whitespace_to_bert[aspect_id]
                                for k, bert_aspect_id in enumerate(bert_aspect_ids):
                                    aspect.append(bert_aspect_id)
                                    if j == 0:
                                        if k == 0:
                                            final_triplet['aspect'] += bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                        else:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                    else:
                                        if k == 0:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                        else:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                    if j == (len(aspect_ids)-1) and k == (len(bert_aspect_ids)-1):
                                        final_triplet['aspect_tags'][bert_aspect_id] = 2
                            if sentiment == 'POS':
                                s_I = 1
                                s_B = 4
                            elif sentiment == 'NEG':
                                s_I = 2
                                s_B = 5
                            elif sentiment == 'NEU':
                                s_I = 3
                                s_B = 6
                            else:
                                print('sentiment error:', sentiment)
                            for j, opinion_id in enumerate(opinion_ids):
                                bert_opinion_ids = whitespace_to_bert[opinion_id]
                                for k, bert_opinion_id in enumerate(bert_opinion_ids):
                                    opinion.append(bert_opinion_id)
                                    if j == 0:
                                        if k == 0:
                                            final_triplet['opinion'] += bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = s_I
                                        else:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = s_I
                                    else:
                                        if k == 0:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = s_I
                                        else:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = s_I
                                    if j == (len(opinion_ids)-1) and k == (len(bert_opinion_ids)-1):
                                        final_triplet['opinion_tags'][bert_opinion_id] = s_B
                            bert_opinion_tail_id = bert_opinion_id
                            r_set[str(aspect_ids)] = final_triplet
                            gold_labels.append((aspect, opinion, sentiment))
                        for k in r_set.keys():
                            all_triplets.append(r_set[k])
                        
                        #汇总的数据
                        length = len(all_triplets[0]['aspect_tags'])
                        TopActions = [0 for i in range(length)]
                        BotOpinionActions = [[] for i in range(length)]
                        
                        for label in all_triplets:
                            asp, opi = label['aspect_tags'], label['opinion_tags']
                            for i in range(length):
                                if asp[i] != 0:
                                    TopActions[i] = asp[i]
                                if asp[i] >= 2:
                                    BotOpinionActions[i] = opi
                        all_triplets_1 = (TopActions, BotOpinionActions)

                        self.data[name].append({'sentext': sentence, 'triplets': all_triplets_1, 'tri': all_triplets, 'labels': gold_labels, 'pos_tags': bert_pos_tags, 'bert_to_whitespace': bert_to_whitespace, 'whitespace_tokens': whitespace_tokens})
                    else:
                        # triplets are not needed for inference
                        self.data[name].append({'sentext': sentence, 'triplets': None, 'tri': None, 'labels': gold_labels, 'pos_tags': bert_pos_tags, 'bert_to_whitespace': bert_to_whitespace, 'whitespace_tokens': whitespace_tokens})
                fl.close()

        # convert POS tags to IDs
        for name in names:
            for item in self.data[name]:
                item['pos_tags'] = ([self.all_pos_tags.index(j) for j in item['pos_tags']])
        
        self.sentiments = ['POS', 'NEG', 'NEU']
        self.sent_count = len(self.sentiments)
        print(self.sentiments)

