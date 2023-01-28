
import pickle
import os
import json
import torch
from torch.utils.data.dataset import Dataset
import sys
from tqdm import tqdm
from transformers import BertTokenizer, BartTokenizer

class Batch():
    def __init__(self,
            inputs,
            input_mask,
            reply_inputs,
            reply_mask,
            reply_labels,
            doc,
            doc_mask,
            graph,
            graph_mask,
            head_text,
            rel_text,
            tail_text,
            ) -> None:
        
        self.inputs = torch.LongTensor(inputs)
        self.input_mask = torch.LongTensor(input_mask)

        self.reply_inputs = torch.LongTensor(reply_inputs)
        self.reply_mask = torch.LongTensor(reply_mask)
        self.reply_labels = torch.LongTensor(reply_labels)

        self.doc = torch.LongTensor(doc)
        self.doc_mask = torch.LongTensor(doc_mask)

        self.graph = torch.LongTensor(graph)
        self.graph_mask = torch.LongTensor(graph_mask)
        
        self.head_text = torch.LongTensor(head_text)
        self.rel_text = torch.LongTensor(rel_text)
        self.tail_text = torch.LongTensor(tail_text)



    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.input_mask = self.input_mask.to(device)
        
        self.reply_inputs = self.reply_inputs.to(device)
        self.reply_labels = self.reply_labels.to(device)
        self.reply_mask = self.reply_mask.to(device)

        self.doc = self.doc.to(device)
        self.doc_mask = self.doc_mask.to(device)

        self.graph = self.graph.to(device)
        self.graph_mask = self.graph_mask.to(device)

        self.head_text = self.head_text.to(device)
        self.rel_text = self.rel_text.to(device)
        self.tail_text = self.tail_text.to(device)



class KBDataset(Dataset):
    def __init__(self,
                name = "dataset",
                type = "train",
                file_path = None) -> None:
        
        super().__init__()
        assert type in ['train', 'dev', 'test'], "Dataset type must be 'train', 'dev' or 'test'"
        
        self.file_path = file_path
        self.type = type
        self.name = name
        self.inputs = []
        self.reply_inputs = []
        self.reply_labels = []
        self.doc = []
        self.tuples = []
        
        self.head_text = []
        self.rel_text = []
        self.tail_text = []
        self.len = 0
        

    def __getitem__(self, index):
        #return super().__getitem__(index)
        return self.inputs[index], self.reply_inputs[index], self.reply_labels[index], self.doc[index], self.tuples[index], self.head_text[index], self.rel_text[index], self.tail_text[index]

    def __len__(self):
        return self.len
    
    def make(self, tokenizer, ent2id, rel2id, src_dir = '.', max_len = None):
        #initialize a new dataset
        path = os.path.join(src_dir, f"{self.type}.json") if self.file_path is None else self.file_path
        assert os.path.exists(path), f"file {path} doesn't exist."

        with open(path, 'r', encoding = 'utf8') as f:
            cases = json.load(f)
        
        #self.inputs = [x["input"] for x in cases]
        #prepare
        sos_prefix = tokenizer.encode("[SOS]", add_special_tokens = False)
        eos_suffix = tokenizer.encode("[EOS]", add_special_tokens = False)
        for turn in tqdm(cases):
            for i in range(len(turn)):
                #print(cases[i])
                input_str = ""
                turn_tuples = []
                #doc_set = []
                doc_str = ""
                if i == 0:
                    input_str = "[SEP]" + turn[i]["input"]
                    turn_tuples = turn[i]["tuples"]
                    #doc_set.append(turn[i]["doc"])
                    doc_str = "[SEP]" + turn[i]["doc"]
                elif i == 1:

                    input_str = turn[i - 1]["input"] + "[HIS]" + turn[i - 1]["reply"] + "[SEP]" + turn[i]["input"]
                    #if max_len is not None and len(input_str) > max_len:
                    #    input_str = "[SEP]" + turn[i]["input"]

                    turn_tuples = turn[i - 1]["tuples"] + turn[i]["tuples"]
                    #doc_set.append(turn[i]["doc"])
                    #if max_len is None or len(turn[i]["doc"]) + len(turn[i - 1]["doc"]) + 2 <= max_len:
                    #    doc_set.append(turn[i - 1]["doc"])
                    doc_str = turn[i - 1]["doc"] + "[SEP]" + turn[i]["doc"]


                else:
                    input_str = turn[i - 2]["input"] + "[HIS]" + turn[i - 2]["reply"] + "[HIS]" + turn[i - 1]["input"] + "[HIS]" + turn[i - 1]["reply"] + "[SEP]" + turn[i]["input"]
                    #if len(input_str) > max_len:
                    #    input_str = turn[i - 1]["input"] + "[HIS]" + turn[i - 1]["reply"] + "[HIS]" + turn[i]["input"]

                    turn_tuples = turn[i - 2]["tuples"] + turn[i - 1]["tuples"] + turn[i]["tuples"]
                    #doc_set.append(turn[i]["doc"])
                    #doc_set.append(turn[i - 1]["doc"])
                    #if max_len is None or len(turn[i - 1]["doc"]) + len(turn[i]["doc"]) + len(turn[i - 1]["doc"]) + 3 <= max_len:
                    #    doc_set.append(turn[i - 2]["doc"])
                    doc_str = turn[i - 2]["doc"] + "[HIS]" + turn[i - 1]["doc"] + "[SEP]" + turn[i]["doc"]

                #self.input_token_type_ids.append([0] * (len(input_str) - len(turn[i]["input"])) + [1] * len(turn[i]["input"]))

                tps = []

                head_text = []
                rel_text = []
                tail_text = []

                vis = {}


                for tuple in turn_tuples:
                    key = "{}-{}-{}".format(tuple[0], tuple[1], tuple[2])
                    if vis.get(key, False):
                        continue
                    tps.append([ent2id[tuple[0]], rel2id[tuple[1]], ent2id[tuple[2]]])
                    head_text.append(tokenizer.encode(tuple[0], add_special_tokens = False))

                    rel_text.append(tokenizer.encode(tuple[1], add_special_tokens = False))
                    tail_text.append(tokenizer.encode(tuple[2], add_special_tokens = False))
                    if len(head_text[-1]) >= 512 or len(rel_text[-1]) >= 512 or len(tail_text[-1]) >= 512:
                        print("find")

                    vis[key] = True
                #print(head_text)
                self.head_text.append(head_text)
                self.rel_text.append(rel_text)
                self.tail_text.append(tail_text)

                self.tuples.append(tps)
                input_ids = tokenizer.encode(input_str, add_special_tokens = False)
                if max_len is not None and len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                self.inputs.append(input_ids)

                #doc_str = "[SEP]".join(list(set(doc_set)))
                #doc_str += "[NOTHING]"
                #self.doc_token_type_ids.append([0] * (len(doc_str) - len(turn[i]["doc"])) + [1] * len(turn[i]["doc"]))
                doc_ids = tokenizer.encode(doc_str, add_special_tokens = False)
                if max_len is not None and len(doc_ids) > max_len:
                    doc_ids = doc_ids[:max_len]

                self.doc.append(doc_ids)

                assert len(self.doc[-1]) <= 512
            for x in turn:
                reply_ids = tokenizer.encode(x["reply"], add_special_tokens = False)
                if max_len is not None and len(reply_ids) + 1 > max_len:
                    reply_ids = reply_ids[: max_len - 1]

                self.reply_inputs.append(sos_prefix + reply_ids)
                self.reply_labels.append(reply_ids + eos_suffix)

            #self.relpy_inputs = [tokenizer.encode("[SOS]" + x["reply"], add_special_tokens = False) for x in cases]
            #self.reply_labels = [tokenizer.encode(x["reply"] + "[EOS]", add_special_tokens = False) for x in cases]
        
        #self.doc = [tokenizer.encode(x["doc"], add_special_tokens = False) for x in cases]
        
        self.len = len(self.inputs)
    

class MyDataset():
    def __init__(self, name, tokenizer, save_path = ".", src_dir = '.', train_set = None, val_set = None, test_set = None, test_unseen_set = None, graph_train_set = None, model_url = 'bert-base-chinese', has_unseen_test_set = True, max_len = None) -> None:
        
        self.name = name
        self.save_path = save_path
        self.src_dir = src_dir
        
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.test_unseen_set = test_unseen_set

        self.has_unseen_test_set = has_unseen_test_set
        

        self.graph_train_set = graph_train_set
        self.model_url = model_url

        if max_len is None:
            #self.max_len = 512 if "bert" in self.model_url.lower() else None
            if "bert" in self.model_url.lower():
                self.max_len = 512
            elif "bart" in self.model_url.lower():
                self.max_len = 1024
            else:
                self.max_len = None
        else:
            self.max_len = max_len

        ''''
        if "bert" in model_url.lower():
            self.tokenizer = BertTokenizer.from_pretrained(model_url)
        elif "bart" in model_url.lower():
            self.tokenizer = BartTokenizer.from_pretrained(model_url)
        else:
            raise f"invalid url '{model_url}'"
        '''
        self.tokenizer = tokenizer

        self.tokenizer.add_special_tokens(
            {'pad_token': '[PAD]',
             'bos_token': '[SOS]', 
             'eos_token': '[EOS]', 
             'unk_token': '[UNK]', 
             'sep_token': '[SEP]',
             'cls_token': '[CLS]',
             'mask_token': '[MASK]',
             })
        self.tokenizer.add_tokens(["[HIS]", "[NOTHING]"])


        self.pad_token_id = self.tokenizer.encode('[PAD]', add_special_tokens = False)[0]
        self.sos_token_id = self.tokenizer.encode('[SOS]', add_special_tokens = False)[0]
        self.eos_token_id = self.tokenizer.encode('[EOS]', add_special_tokens = False)[0]
        self.sep_token_id = self.tokenizer.encode('[SEP]', add_special_tokens = False)[0]
        self.his_token_id = self.tokenizer.encode('[HIS]', add_special_tokens = False)[0]
        self.nothing_token_id = self.tokenizer.encode('[NOTHING]', add_special_tokens = False)[0]
        #self.

    

        self.num_token = len(self.tokenizer.get_vocab())
    
    def from_preprocessed(self, path):
        #path = os.path.join(src_dir, self.save_name + ".pkl")
        assert os.path.exists(path), f"cache file {path} doesn't exist."

        with open(path, 'rb') as f:
            data_dic = pickle.load(f)
            self.name = data_dic["name"]
            self.train_set = data_dic["train_set"]
            self.val_set = data_dic["val_set"]
            self.test_set = data_dic["test_set"]
            self.test_unseen_set = data_dic["test_unseen_set"]
            self.graph_train_set = data_dic["graph_set"]
        
    
    def make(self, save_cache = True):
        print("Preparing ConvE Training Set...")
        self.graph_train_set = GraphDataset(self.name)
        self.graph_train_set.make(dir = self.src_dir)
        print("Preparing Training Set...")
        #prepare training set
        self.train_set = KBDataset(self.name, "train")
        self.train_set.make(tokenizer = self.tokenizer, ent2id = self.graph_train_set.ent2id, rel2id = self.graph_train_set.rel2id, src_dir = self.src_dir, max_len = self.max_len)
        print("Preparing Validating Set...")
        #prepare validating set
        self.val_set = KBDataset(self.name, "dev")
        self.val_set.make(tokenizer = self.tokenizer, ent2id = self.graph_train_set.ent2id, rel2id = self.graph_train_set.rel2id, src_dir = self.src_dir, max_len = self.max_len)
        print("Preparing Test Set...")
        #prepare testing set
        self.test_set = KBDataset(self.name, "test")
        self.test_set.make(tokenizer = self.tokenizer, ent2id = self.graph_train_set.ent2id, rel2id = self.graph_train_set.rel2id, src_dir = self.src_dir, max_len = self.max_len)


        if self.has_unseen_test_set:
            print("Preparing Test Set(Unseen)...")
            self.test_unseen_set = KBDataset(self.name, "test", file_path = os.path.join(self.src_dir, "test_unseen.json"))
            self.test_unseen_set.make(tokenizer = self.tokenizer, ent2id = self.graph_train_set.ent2id, rel2id = self.graph_train_set.rel2id, src_dir = self.src_dir, max_len = self.max_len)

        if save_cache:
            print("Saving Cache...")
            self.save()
        
        print("Done.")
        
    def save(self):
        #path = os.path.join(src_dir if save_dir is None else save_dir, self.save_name + ".pkl") 
        path = self.save_path
        data_dic = {
            "name": self.name,
            "train_set": self.train_set,
            "val_set": self.val_set,
            "test_set": self.test_set,
            "test_unseen_set": self.test_unseen_set,
            "graph_set": self.graph_train_set
        }
        with open(path, 'wb') as f:
            pickle.dump(data_dic, f)
    
    def padding3D(self, data, padding_object):
        max_lines = 0
        max_token_num = 0
        
        for mat in data:
            max_lines = max(max_lines, len(mat))
            for line in mat:
                max_token_num = max(max_token_num, len(line))
        
        new_data = []
        for mat in data:
            new_mat = []
            ln = len(mat)

            for line in mat:
                l = len(line)
                #print(line)
                new_mat.append(line + [padding_object] * (max_token_num - l))
            
            for i in range(max_lines - ln):
                new_mat.append([padding_object] * max_token_num)

            new_data.append(new_mat)
        
        return new_data

    def padding(self, data, padding_object):
        max_len = max([len(x) for x in data])
        new_data = []
        masks = [] #1 for not masked
        for x in data:
            l = len(x)
            mask = [1] * l + [0] * (max_len - l)
            new_x = x + [padding_object] * (max_len - l)
            new_data.append(new_x)
            masks.append(mask)
        
        return new_data, masks

    def collate_fn(self, data) -> Batch:
        #print(len(data))
        inputs = [x[0] for x in data]
        reply_inputs = [x[1] for x in data]
        reply_labels = [x[2] for x in data]
        doc = [x[3] for x in data]
        tps = [x[4] for x in data]
        head_text = [x[5] for x in data]
        rel_text = [x[6] for x in data]
        tail_text = [x[7] for x in data]

        #print(head_text)
        inputs, input_mask = self.padding(inputs, self.pad_token_id)
        reply_inputs, reply_input_mask = self.padding(reply_inputs, self.pad_token_id)
        reply_labels, reply_labels_mask = self.padding(reply_labels, -100)
        doc, doc_mask = self.padding(doc, self.pad_token_id)
        tps, tp_mask = self.padding(tps, [0, 0, 0])

        head_text = self.padding3D(head_text, 0)
        
        rel_text = self.padding3D(rel_text, 0)
        tail_text = self.padding3D(tail_text, 0)


        #return self.tokenizer(inputs, return_tensors = 'pt', padding = True), self.tokenizer(reply_inputs, return_tensors = 'pt', padding = True), self.tokenizer(reply_labels, return_tensors = 'pt', padding = True)
        return Batch(
            inputs = inputs,
            input_mask = input_mask,
            reply_inputs=reply_inputs,
            reply_mask=reply_input_mask,
            reply_labels=reply_labels,
            doc = doc,
            doc_mask = doc_mask,
            graph = tps,
            graph_mask = tp_mask,
            head_text = head_text,
            rel_text = rel_text,
            tail_text = tail_text,
        )


class GraphDataset(Dataset):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.tuples = []
        self.ent_num = None
        self.rel_num = None
        self.ent2id = None
        self.rel2id = None
        #self.init()

    def __getitem__(self, index):
        return self.heads[index], self.rels[index], self.tails[index]
    
    def __len__(self):
        return len(self.tuples)
    
    def make(self, dir):
    
        with open(dir + '/ents.json', 'r', encoding = "utf8") as f:
            ents = json.load(f)
        
        with open(dir + '/rels.json', 'r', encoding = "utf8") as f:
            rels = json.load(f)
        
        self.ent2id = {v: k + 1 for k, v in enumerate(ents)}
        self.ent2id["[NOTHING]"] = 0

        self.ent_num = len(self.ent2id.keys())
        
        self.rel2id = {v: k + 1 for k, v in enumerate(rels)}
        self.rel2id["[NOTHING]"] = 0

        self.rel_num = len(self.rel2id.keys())

        with open(dir + f'/kb_{self.name}.json', 'r', encoding = "utf8") as f:
            data = json.load(f)
            for key in data.keys():
                for tp in data[key]:
                    if tp[1] == "Information" or tp[2] == "":
                        continue
                    self.tuples.append((self.ent2id[tp[0]], self.rel2id[tp[1]], self.ent2id[tp[2]]))

if __name__ == "__main__":
    #tokenizer = BertTokenizer.from_pretrained('facebook/bart-base')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    ds = MyDataset(name = "wow", save_path = "/root/autodl-tmp/processed_data/wow-bart-base.pkl", tokenizer = tokenizer, src_dir = "/root/autodl-tmp/processed_data/wow")
    #ds = MyDataset(name = "wow", save_path = "/root/autodl-tmp/processed_data/film-bert-base.pkl", tokenizer = tokenizer, src_dir = "/root/autodl-tmp/processed_data/film")
    #ds = MyDataset()
    ds.make()
    #ds.from_preprocessed('/data/yangpan/processed_data/film-new.pkl')
    #print(ds.tokenizer.decode(ds.train_set.reply_labels[-1]))
    #print(ds.nothing_token_id)
    print(ds.tokenizer.decode(ds.train_set.inputs[-1]))
    print(ds.tokenizer.decode(ds.train_set.reply_inputs[-1]))
    print(ds.tokenizer.decode(ds.train_set.reply_labels[-1]))
    print(ds.tokenizer.decode(ds.train_set.doc[-1]))
    #print(ds.train_set.input_token_type_ids[-1])
    #print(ds.train_set.doc_token_type_ids[-1])