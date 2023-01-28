import json
import os

class Config():
    def __init__(self) -> None:
        self.paras = {}
    
    def read_json_config(self, path):
        assert os.path.exists(path), f'Json config file {path} doesn\'t exist'
        with open(path, 'r', encoding = 'utf8') as f:
            paras = json.load(f)
        print('-' * 20)
        for key in self.paras.keys():
            assert key in paras.keys() or self.paras[key] is not None, f'Missing required parameters "{key}".'
            
            if key not in paras.keys():
                #print(f"Using default value: '{key}': {self.paras[key]}.")
                continue

            self.paras[key] = paras[key]
            print(f"'{key}': {self.paras[key]}.")    
    
    def dump_json(self, path):
        with open(path, 'w', encoding = 'utf8') as f:
            json.dump(self.paras, f, ensure_ascii = False)
    
    def __getitem__(self, key):
        return self.paras[key]
    
    def add_paras(self, key, default_value = None):
        self.paras[key] = default_value

    
class TrainConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        
        self.add_paras("lr")
        self.add_paras("epochs")
        self.add_paras("warm_up_steps")
        self.add_paras("do_eval", False)
        self.add_paras("batch_size")
        self.add_paras("save_path")
        self.add_paras("device")
        self.add_paras("checkpoint_path", ".")
        self.add_paras("seed", 114514)
        self.add_paras("start_from_checkpoint", False)
        self.add_paras("pin_memory", False)
        self.add_paras("eval_result_save_path", './res.json')
        self.add_paras("num_workers", 2)

class EvalConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.add_paras("batch_size", 32)
        self.add_paras("device")
        self.add_paras("checkpoint_path")
        self.add_paras("pin_memory", False)
        self.add_paras("has_unseen_test_set", True)
        self.add_paras("test_seen_result_save_path", "./res_seen.json")
        self.add_paras("test_unseen_result_save_path", "./res_unseen.json")
        self.add_paras("num_workers", 2)
        self.add_paras("max_len", 64)
        self.add_paras("test_seen_raw_text_save_path", './res_seen_raw.json')
        self.add_paras("test_unseen_raw_text_save_path", './res_unseen_raw.json')




class ModelConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.add_paras("model_name", "model")
        self.add_paras("model_dim", 768)
        self.add_paras("hidden_dim", 1024)
        self.add_paras("model_url")
        self.add_paras("ent_dim", 768)
        self.add_paras("rel_dim", 768)
        self.add_paras("nhead", 8)
        self.add_paras("drop_out", 0.1)
        self.add_paras("mix_encoder_layer_num", 2)
        self.add_paras("mix_decoder_layer_num", 2)
        self.add_paras("graph_attn_method", "mean-A")
        self.add_paras("do_graph_fusion", True)
        self.add_paras("do_doc_fusion", True)
        self.add_paras("pretrained_weights_cache_dir", ".")
        self.add_paras("back_bone", "bert")
        self.add_paras("do_copy", True)

class RunConfig():
    def __init__(self, train_config:Config = None, eval_config:Config = None) -> None:
        
        if train_config is None:
            self.train_config = TrainConfig()
        else:
            self.train_config = train_config     
        
        if eval_config is None:
            self.eval_config = EvalConfig()
        else:
            self.eval_config = eval_config
        
    def read_json_config(self, train_config_path, eval_config_path, use_the_same_checkpoint = False):
        self.train_config.read_json_config(train_config_path)
        self.eval_config.read_json_config(eval_config_path)
        if use_the_same_checkpoint:
            if self.train_config["checkpoint_path"] != self.eval_config["checkpoint_path"]:
                print("You chose to use the same checkpoint but the checkpoint in the train config is '{}' while the checkpoint in the eval config is '{}'".format(self.train_config["checkpoint_path"]), self.eval_config["checkpoint_path"])
                self.eval_config.paras["checkpoint_path"] = self.train_config["checkpoint_path"]
                print("Changed '{}' to '{}'".format(self.eval_config["checkpoint_path"], self.train_config["checkpoint_path"]))
                