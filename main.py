import torch
import random
import numpy as np
from Config import *
from Dataset import *
from Model import Model
from torch.utils.data import DataLoader
from Evaluation import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def train(model, datasets, config: TrainConfig):
    if config["seed"] is not None:
        set_seed(config["seed"])
    
    ds = datasets
    model = model.to(config["device"])
    
    if config["start_from_checkpoint"]:
        model.load_checkpoints(config["checkpoint_path"])
    
    loader = DataLoader(dataset=ds.train_set, batch_size=run_config["batch_size"], shuffle=True, num_workers=config["num_workers"],
                        collate_fn=ds.collate_fn,
                        pin_memory = config["pin_memory"])
    
    model.train_self(dataloader = loader, config = config)

def eval(model, datasets, config: EvalConfig):

    model = model.to(config["device"])
    model.load_checkpoints(config["checkpoint_path"])

    loader = DataLoader(dataset=datasets.test_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
                        collate_fn=ds.collate_fn, pin_memory = config["pin_memory"])
    qs, gs, gens = model.gen_self(loader, config["device"], max_len = config["max_len"])

    #print(datasets.tokenizer.decode(gen[-1]))

    result_seen = {
        "query": qs,
        "gold": gs,
        "gen": gens
    }
    #print("-----Seen------")
    #ress: EvaluationrRes() = calculate_metrics()
    with open(config["test_seen_result_save_path"], 'w', encoding = 'utf8') as fo:
        json.dump(result_seen, fo, ensure_ascii = False)

    if config["has_unseen_test_set"]:
        loader = DataLoader(dataset=datasets.test_unseen_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
                    collate_fn=ds.collate_fn, pin_memory = config["pin_memory"])
        qs, gs, gens = model.gen_self(loader, config["device"], max_len = config["max_len"])

        #print(datasets.tokenizer.decode(gen[-1]))

        result_unseen = {
            "query": qs,
            "gold": gs,
            "gen": gens
        }

        with open(config["test_unseen_result_save_path"], 'w', encoding = 'utf8') as fo:
            json.dump(result_unseen, fo, ensure_ascii = False)
    
    print("Done")

def calculate_metrics(golds, gens, tokenizer):
    res = EvaluationrRes()
    print("Calculate BLEU...")
    res.bleu1, res.bleu2, res.bleu3, res.bleu4 = calculate_bleu(references= [[x] for x in golds], hypothesis = gens)
    print("Calculate DISTINCT...")
    res.distinct1, res.distinct2 = calculate_distinct(gens)
    ref = [" ".join([str(x) for x in y]) for y in golds]
    hyp = [" ".join([str(x) for x in y]) if y != [] else "[NOTHING]"  for y in gens]
    #print(hyp)
    print("Calculate ROUGE...")
    res.rouge1, res.rouge2, res.rougeL = calculate_rouge(hypothesis=hyp, references=ref)
    
    gold_sents = [tokenizer.decode(x) for x in golds]
    #print(gold_sents)
    ref_sents = [gold_sents]
    hyp_sents = [tokenizer.decode(x) if x != [] else "[NOTHING]" for x in gens] #In case of empty sentences 
    #print(len(ref_sents), len(hyp_sents))
    print("Calculate F1...")
    res.F1 = calculate_F1(references = [[x] for x in gold_sents], hypothesis = hyp_sents)
    print("Calculate METEOR and embedding-based metrics...")
    res.meteor ,res.AVG, res.EXT, res.GREEDY = calculate_meteor_and_embedding_based_metrics(hypothesis = hyp_sents, references = ref_sents)
    
    return res

if __name__ == "__main__":    
    config = ModelConfig()
    config.read_json_config('./settings/model_config.json')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    ds = MyDataset(name = "film", tokenizer = tokenizer)
    ds.from_preprocessed('/root/autodl-tmp/processed_data/wow-bert-base-uncase.pkl')
    #ds.from_preprocessed('/root/autodl-tmp/processed_data/film-bert-base.pkl')
    special_ids = {
        "[SOS]": ds.sos_token_id,
        "[EOS]": ds.eos_token_id,
        "[PAD]": ds.pad_token_id,
        "[HIS]": ds.his_token_id,
        "[SEP]": ds.sep_token_id
    }
    print(special_ids)

    md = Model(
        token_num=ds.num_token,
        ent_num=ds.graph_train_set.ent_num,
        rel_num=ds.graph_train_set.rel_num,
        special_token_ids = special_ids,
        config=config
    )
    #md.load_graph_embedding_weights('/root/autodl-tmp/checkpoints/wow_conve')

    #Randomly Initialize
    md.load_graph_embedding_weights('/root/autodl-tmp/weights/wow')

    run_config = TrainConfig()
    run_config.read_json_config("./settings/train_config.json")

    train(md, ds, run_config)

    eval_config = EvalConfig()
    eval_config.read_json_config("./settings/eval_config.json")
    eval(datasets = ds, config = eval_config, model = md)

    with open(eval_config["test_seen_result_save_path"], 'r', encoding = 'utf8') as fs, open(eval_config["test_unseen_result_save_path"], 'r', encoding = 'utf8') as fu:
        result_seen = json.load(fs)
        result_unseen = json.load(fu)

    #for query, gold, gen in zip(res["query"], res["gold"], res['gen']):
    #    print("[query]:" + ds.tokenizer.decode(query))
    #    print("[gen]:" + ds.tokenizer.decode(gen))
    #    print("[gold]:" + ds.tokenizer.decode(gold))
    
    ress: EvaluationrRes = calculate_metrics(result_seen["gold"], result_seen["gen"], ds.tokenizer)
    print("-----Seen-----")
    ress.show_results()

    raw_text_seen = []
    for query, gold, gen in zip(result_seen["query"], result_seen["gold"], result_seen['gen']):
        
        #print("[query]:" + ds.tokenizer.decode(query))
        #print("[gen]:" + ds.tokenizer.decode(gen))
        #print("[gold]:" + ds.tokenizer.decode(gold))

        raw_text_seen.append(
            {
                "query": ds.tokenizer.decode(query),
                "gen": ds.tokenizer.decode(gen),
                "gold": ds.tokenizer.decode(gold)
            }
        )
    
    with open(eval_config["test_seen_raw_text_save_path"], 'w', encoding = 'utf8') as f:
        json.dump(raw_text_seen, f, ensure_ascii = False)

    
    resu: EvaluationrRes = calculate_metrics(result_unseen["gold"], result_unseen["gen"], ds.tokenizer)
    print("-----Unseen-----")
    resu.show_results()

    raw_text_unseen = []

    for query, gold, gen in zip(result_unseen["query"], result_unseen["gold"], result_unseen['gen']):
        
        #print("[query]:" + ds.tokenizer.decode(query))
        #print("[gen]:" + ds.tokenizer.decode(gen))
        #print("[gold]:" + ds.tokenizer.decode(gold))

        raw_text_unseen.append(
            {
                "query": ds.tokenizer.decode(query),
                "gen": ds.tokenizer.decode(gen),
                "gold": ds.tokenizer.decode(gold)
            }
        )
    
    with open(eval_config["test_unseen_raw_text_save_path"], 'w', encoding = 'utf8') as f:
        json.dump(raw_text_unseen, f, ensure_ascii = False)

