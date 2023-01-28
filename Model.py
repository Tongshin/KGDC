

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertLMHeadModel, BertConfig, get_scheduler, BartModel, BartForConditionalGeneration

from Dataset import *
from FusionDecoder import FusionDecoder
from FusionEncoder import FusionEncoder
from Config import ModelConfig, TrainConfig


class Model(nn.Module):

    def __init__(self, token_num, ent_num, rel_num, special_token_ids: dict, config:ModelConfig = None) -> None:
        super().__init__()
        
        self.config = config
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.token_num = token_num

        self.special_token_ids = special_token_ids
        if special_token_ids.get("[PAD]", None) is not None:
            self.pad_id = special_token_ids.get("[PAD]", None)

        if special_token_ids.get("[SOS]", None) is not None:
            self.sos_id = special_token_ids.get("[SOS]", None)
        if special_token_ids.get("[EOS]", None) is not None:
            self.eos_id = special_token_ids.get("[EOS]", None)
        if special_token_ids.get("[SEP]", None) is not None:
            self.sep_id = special_token_ids.get("[SEP]", None)
        if special_token_ids.get("[HIS]", None) is not None:
            self.his_id = special_token_ids.get("[HIS]", None)

        #Embeddings
        self.ent_embedding = nn.Embedding(ent_num + 1, config.paras["ent_dim"])
        self.rel_embedding = nn.Embedding(rel_num + 1, config.paras["rel_dim"])
        #self.input_seg_embedding = nn.Embedding(8, config["model_dim"])
        #self.doc_seg_embedding = nn.Embedding(8, config["model_dim"])

        #Attention module for copy mechanism
        self.tail_text_attn = nn.MultiheadAttention(embed_dim = config["model_dim"], num_heads=config["nhead"], dropout = config["drop_out"], batch_first = True)

        #Basic encoder
        if self.config["back_bone"] == "bart":
            self.bart_c = BartForConditionalGeneration.from_pretrained(config["model_url"], cache_dir = config["pretrained_weights_cache_dir"])
            self.bart = self.bart_c.model
            self.pos_embedding = nn.Embedding(num_embeddings = 1024, embedding_dim = config["model_dim"])
            self.doc_bart = BartForConditionalGeneration.from_pretrained(config["model_url"], cache_dir = config["pretrained_weights_cache_dir"])
            self.doc_encoder = self.doc_bart.model.encoder

        if self.config["back_bone"] == "bert":
            self.text_encoder = BertModel.from_pretrained(config["model_url"], cache_dir = config["pretrained_weights_cache_dir"], mirror = "tuna")

        elif self.config["back_bone"] == "bart":
            self.text_encoder = self.bart.encoder
        else:
            raise f'invalid back bone type {self.config["back_bone"]}'

        #Fusion Encoder
        self.mix_encoder = FusionEncoder(
            embedding_dim = config["model_dim"],
            hidden_dim = config["hidden_dim"],
            nhead = config["nhead"],
            drop_out = config["drop_out"],
            nlayer = config["mix_encoder_layer_num"],
            graph_attn_method = config["graph_attn_method"],
            use_doc_attn = config["do_doc_fusion"],
            use_graph_attn = config["do_graph_fusion"]
        )

        #Basic decoder
        if self.config["back_bone"] == "bert":
            self.basic_decoder = BertLMHeadModel.from_pretrained(config["model_url"], is_decoder = True, add_cross_attention = True, cache_dir = config["pretrained_weights_cache_dir"])
        elif self.config["back_bone"] == "bart":
            self.basic_decoder = self.bart.decoder

        #Vocabulary
        self.resize_embedding(self.token_num)

        #Fusion Decoder
        self.mix_decoder = FusionDecoder(
            embedding_dim = config["model_dim"],
            hidden_dim = config["hidden_dim"],
            nhead = config["nhead"],
            drop_out = config["drop_out"],
            nlayer = config["mix_decoder_layer_num"],
            graph_attn_method = config["graph_attn_method"],
            use_doc_attn = config["do_doc_fusion"],
            use_graph_attn = config["do_graph_fusion"]
        )

        #self.tail_text_attn = nn.MultiheadAttention(embed_dim = config["model_dim"], num_heads = config["nhead"], dropout = config["drop_out"], batch_first = True)

        #Linear
        self.linear = nn.Linear(config.paras["model_dim"], self.token_num)

        #Normalization
        self.norm1 = nn.LayerNorm(config["model_dim"])
        self.norm2 = nn.LayerNorm(config["model_dim"])
        self.norm3 = nn.LayerNorm(config["model_dim"])
        #self.norm4 = nn.LayerNorm(config["model_dim"])
        #self.norm5 = nn.LayerNorm(config["model_dim"])


        #dropout for text embeddings
        #self.drop1 = nn.Dropout(config["drop_out"])
        #self.drop2 = nn.Dropout(config["drop_out"])

        #P_pointer
        if self.config["do_doc_fusion"] and self.config["do_graph_fusion"]:
            self.p_pointer = nn.Sequential(
                nn.Linear(self.token_num, 4),
                nn.Softmax(dim = -1)
            )
        elif self.config["do_doc_fusion"] or self.config["do_graph_fusion"]:
            self.p_pointer = nn.Sequential(
                nn.Linear(self.token_num, 3),
                nn.Softmax(dim = -1)
            )
        else:
            self.p_pointer = nn.Sequential(
                nn.Linear(self.token_num, 2),
                nn.Softmax(dim = -1)
            )
    
    def load_graph_embedding_weights(self, dir):
        self.ent_embedding.load_state_dict(torch.load(os.path.join(dir, "ent_embed.pkl"), map_location = "cpu"),)
        self.rel_embedding.load_state_dict(torch.load(os.path.join(dir, "rel_embed.pkl"), map_location = "cpu"))


    def train_self(self, dataloader: DataLoader, config:TrainConfig):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr = config.paras["lr"])
        criterion = nn.NLLLoss()

        scheduler = get_scheduler(
            name = 'linear',
            optimizer = optimizer,
            num_warmup_steps = config.paras["warm_up_steps"],
            num_training_steps = config.paras["epochs"] * len(dataloader)
        )
        for ep in range(config.paras["epochs"]):
            for i, bacth in enumerate(dataloader):
                bc:Batch = bacth
                bc.to(config["device"])

                optimizer.zero_grad()
                output = self.forward(
                    src = bc.inputs,
                    tgt = bc.reply_inputs,
                    doc = bc.doc,
                    src_mask = bc.input_mask,
                    tgt_mask = bc.reply_mask,
                    doc_mask = bc.doc_mask,
                    graph = bc.graph,
                    graph_mask = bc.graph_mask,
                    head_text = bc.head_text,
                    rel_text = bc.rel_text,
                    tail_text = bc.tail_text
                )
                loss = criterion(output.permute(0, 2, 1), bc.reply_labels)
                loss.backward()
                print(f"Round: {ep + 1}, it: {i + 1}, loss: {loss.item()}.")
                optimizer.step()
                scheduler.step()

            self.save(config["save_path"])
            torch.cuda.empty_cache()
        print("done")


    def save(self, dir = './checkpoints'):
        torch.save(self.state_dict(), dir)

    def generate_self(self):
        self.eval()

    def load_checkpoints(self, path):
        print("Loading checkpoint from {}".format(path))
        self.load_state_dict(torch.load(path))

    def encode(self, src, doc, src_mask, doc_mask, graph, graph_mask):

        #B, N, _ = graph.shape
        head = graph[:,:,0]
        rel = graph[:, :,1]
        tail = graph[:, :, 2]

        encoder_output = self.text_encoder(
            input_ids = src,
            #inputs_embeds = src_embedding,
            token_type_ids = torch.zeros_like(src).masked_fill(src == self.sep_id, 1).cumsum(dim = -1),
            attention_mask = src_mask
        )


        #with torch.no_grad():
        head_embeddings = self.ent_embedding(head)
        rel_embeddings = self.rel_embedding(rel)
        tail_embeddings = self.ent_embedding(tail)

        src_hidden = encoder_output.last_hidden_state
        #print(1, torch.any(torch.isnan(src_hidden)))
        #print(torch.any(torch.isnan(d)))
        if self.config["back_bone"] == "bart":
            doc_output = self.doc_encoder(
                input_ids = doc,
                attention_mask = doc_mask
            )
        elif self.config["back_bone"] == "bert":
            #doc_embedding = self.get_graph_text_embedding(doc)

            doc_output = self.text_encoder(
                input_ids = doc,
                #inputs_embeds = doc_embedding,
                token_type_ids=torch.zeros_like(doc).masked_fill(doc == self.sep_id, 1).cumsum(dim=-1),
                attention_mask = doc_mask
            )

        doc_hidden = doc_output.last_hidden_state
        #print(2, torch.any(torch.isnan(doc_hidden)))
        #mix encoding
        new_src_hidden, new_doc_hidden = self.mix_encoder.forward(
            src = src_hidden,
            doc = doc_hidden,
            src_mask = src_mask,
            doc_mask = doc_mask,
            head = head_embeddings,
            rel = rel_embeddings,
            tail = tail_embeddings,
            graph_mask = graph_mask
        )
        #print(3, torch.any(torch.isnan(new_src_hidden)))
        #print(4, torch.any(torch.isnan(new_doc_hidden)))
        return self.norm1(src_hidden + new_src_hidden), self.norm2(doc_hidden + new_doc_hidden), head_embeddings, rel_embeddings, tail_embeddings
    
    def decode(self, memory, tgt, doc, src_mask, tgt_mask, doc_mask, head_embeddings, rel_embeddings, tail_embeddings, graph_mask, history_text, doc_text, head_text, rel_text, tail_text):
        #rough decoding
        #print(memory.shape, src_mask.shape)
        #print(src_mask.shape)
        #print(tgt.shape, tgt_mask.shape)
        decoder_output = self.basic_decoder(
            encoder_hidden_states = memory,
            encoder_attention_mask = src_mask,
            input_ids = tgt,
            attention_mask = tgt_mask,
            output_hidden_states = True,
            token_type_ids = torch.ones_like(tgt)
        )

        if self.config["back_bone"] == "bert":
            decoder_hidden = decoder_output.hidden_states[-1]

        elif self.config["back_bone"] == "bart":
            decoder_hidden = decoder_output.last_hidden_state

        #print(torch.any(torch.isnan(decoder_hidden)))
        #mix decoding
        new_decoder_hidden, h_attn, g_attn, d_attn = self.mix_decoder.forward(
            memory = memory,
            reply = decoder_hidden,
            doc = doc,
            memory_mask = src_mask,
            reply_mask = tgt_mask,
            doc_mask = doc_mask,
            head = head_embeddings,
            rel = rel_embeddings,
            tail = tail_embeddings,
            graph_mask = graph_mask
        )
        #print(torch.any(torch.isnan(new_decoder_hidden)))
        new_decoder_hidden = self.norm3(decoder_hidden + new_decoder_hidden)
        pred = self.linear(new_decoder_hidden)
        
        if not self.config["do_copy"]:
            return F.log_softmax(pred + 1e-20, dim = -1)

        p_pointer = self.p_pointer(pred)

        #prepare
        B, N, L = tail_text.shape

        #head_text_ = head_text.view(B, -1)
        #rel_text_ = rel_text.view(B, -1)
        tail_text_ = tail_text.view(B, -1)

        #head_text_embeddings = self.get_graph_text_embedding(head_text).view(B, -1, self.config["model_dim"])
        #rel_text_embeddings = self.get_graph_text_embedding(rel_text).view(B, -1, self.config["model_dim"])
        tail_text_embeddings = self.get_graph_text_embedding(tail_text).view(B, -1, self.config["model_dim"])
        
        #print(p_pointer.shape)
        if self.config["do_doc_fusion"] and self.config["do_graph_fusion"]:
            ph = p_pointer[:, :, 0].unsqueeze(-1)
            pd = p_pointer[:, :, 1].unsqueeze(-1)
            pk = p_pointer[:, :, 2].unsqueeze(-1)
            pv = p_pointer[:, :, 3].unsqueeze(-1)
        elif self.config["do_doc_fusion"]:
            ph = p_pointer[:, :, 0].unsqueeze(-1)
            pd = p_pointer[:, :, 1].unsqueeze(-1)
            #pk = p_pointer[:, :, 2].unsqueeze(-1)
            pv = p_pointer[:, :, 2].unsqueeze(-1)
        elif self.config["do_graph_fusion"]:
            ph = p_pointer[:, :, 0].unsqueeze(-1)
            #pd = p_pointer[:, :, 1].unsqueeze(-1)
            pk = p_pointer[:, :, 1].unsqueeze(-1)
            pv = p_pointer[:, :, 2].unsqueeze(-1)
        else:
            ph = p_pointer[:, :, 0].unsqueeze(-1)
            #pd = p_pointer[:, :, 1].unsqueeze(-1)
            #pk = p_pointer[:, :, 2].unsqueeze(-1)
            pv = p_pointer[:, :, 1].unsqueeze(-1)
        #h_attn = self.mix_decoder.layers[-1].history_w
        #d_attn = self.mix_decoder.layers[-1].doc_w
        #g_attn = self.mix_decoder.layers[-1].graph_w
        #print(B, N, L)
        #print(g_attn.shape)
        _, k_attn = self.tail_text_attn.forward(
            query = new_decoder_hidden,
            key = tail_text_embeddings,
            value = tail_text_embeddings,
            key_padding_mask = (tail_text_ == 0)
        )
        #t_attn = F.softmax(torch.mul(k_attn.view(B, -1, N, L), g_attn.unsqueeze(-1)).view(B, -1, N * L), dim = -1)
        #k_attn =
        #print(t_attn.shape)
        
        output = F.softmax(pred, dim = -1)
        output = output * pv
        if self.config["do_doc_fusion"]:
            output = output.scatter_add(2, doc_text.unsqueeze(1).repeat(1, output.shape[1], 1), pd * d_attn)
        
        output = output.scatter_add(2, history_text.unsqueeze(1).repeat(1, output.shape[1], 1), ph * h_attn)

        if self.config["do_graph_fusion"]:
            output = output.scatter_add(2, tail_text_.unsqueeze(1).repeat(1, output.shape[1], 1), pk * k_attn)

        output = output + 1e-20

        return torch.log(output)

    def forward(self, src, tgt, doc, src_mask, tgt_mask, doc_mask, graph, graph_mask, head_text, rel_text, tail_text):
        #encoding
        memory, doc_hidden, head_embeddings, rel_embeddings, tail_embeddings = self.encode(
            src = src,
            doc = doc,
            src_mask = src_mask,
            doc_mask = doc_mask,
            graph = graph,
            graph_mask = graph_mask
        )
        #decoding

        pred = self.decode(
            memory = memory,
            tgt = tgt,
            doc = doc_hidden,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            doc_mask = doc_mask,
            head_embeddings = head_embeddings,
            rel_embeddings = rel_embeddings,
            tail_embeddings = tail_embeddings,
            graph_mask = graph_mask,
            history_text = src,
            doc_text = doc,
            head_text = head_text,
            rel_text = rel_text,
            tail_text = tail_text
        )

        return pred
     
    def resize_embedding(self, size):
        self.text_encoder.resize_token_embeddings(size)
        self.basic_decoder.resize_token_embeddings(size)
        if self.config["back_bone"] == 'bart':
            self.doc_encoder.resize_token_embeddings(size)
    
    def get_graph_text_embedding(self, text): 
        #print(text.shape)
        if self.config['back_bone'] == "bert":
            text_embedding = self.text_encoder.embeddings.word_embeddings(text)
            #pos = torch.ones_like(text).cumsum(dim = -1)
            pos = torch.arange(0, text.shape[-1], 1, dtype=torch.long, device=text.device)
            text_pos_embedding = self.text_encoder.embeddings.position_embeddings(pos)
            seg = torch.zeros_like(text)
            text_seg_embedding = self.text_encoder.embeddings.token_type_embeddings(seg)

            return self.text_encoder.embeddings.dropout(self.text_encoder.embeddings.LayerNorm(text_embedding + text_seg_embedding + text_pos_embedding))
        elif self.config["back_bone"] == "bart":
            text_embedding = self.text_encoder.embed_tokens(text) * self.text_encoder.embed_scale
            pos = torch.arange(0, text.shape[-1], 1, dtype=torch.long, device=text.device)
            pos_embedding = self.pos_embedding(pos)

            embedding = self.text_encoder.layernorm_embedding(text_embedding + pos_embedding)
            return F.dropout(embedding, p = self.text_encoder.dropout, training = self.training)

    def get_input_text_embedding(self, inputs): # Not used
        if self.config["back_bone"] == "bert":
            token_embedding = self.text_encoder.embeddings.word_embeddings(inputs)
            pos = torch.arange(0, inputs.shape[1], 1, dtype=torch.long, device=inputs.device)
            pos_embedding = self.text_encoder.embeddings.position_embeddings(pos)
            seg = torch.zeros_like(inputs)

            seg = seg.masked_fill(inputs == self.sep_id, 1).cumsum(dim = -1)
            #print(seg)
            text_seg_embedding = self.input_seg_embedding(seg)
            return self.drop2(self.norm4(token_embedding + pos_embedding + text_seg_embedding))

    def get_doc_text_embedding(self, doc): # Not used
        if self.config["back_bone"] == "bert":
            token_embedding = self.text_encoder.embeddings.word_embeddings(doc)
            pos = torch.arange(0, doc.shape[1], 1, dtype=torch.long, device=doc.device)
            pos_embedding = self.text_encoder.embeddings.position_embeddings(pos)
            seg = torch.zeros_like(doc)
            seg = seg.masked_fill(doc == self.sep_id, 1).cumsum(dim = -1)
            #print(seg)
            text_seg_embedding = self.doc_seg_embedding(seg)
            return self.drop2(self.norm5(token_embedding + pos_embedding + text_seg_embedding))

    def gen_self(self, dataloader: DataLoader, device, max_len = 64):
        self.eval()

        querys = []
        golds = []
        gens = []

        with torch.no_grad():
            for batch in tqdm(dataloader, unit = 'batch'):
                b: Batch = batch
                b.to(device)
                words = self.gen(b, max_len = max_len)

                gens += words
                querys += b.inputs.cpu().tolist()
                golds += b.reply_labels.cpu().tolist()
        
        return self.purify(querys), self.purify(golds), self.purify(gens)
    
    def gen(self, batch: Batch, max_len = 64):
        graph = batch.graph

        memory, doc_hidden, head_embeddings, rel_embeddings, tail_embeddings = self.encode(
            src = batch.inputs,
            doc = batch.doc,
            src_mask = batch.input_mask,
            doc_mask = batch.doc_mask,
            graph = graph,
            graph_mask = batch.graph_mask
        )
        
        tgt = torch.zeros((batch.inputs.size(0), 1)).long().to(batch.inputs.device)
        outputs = torch.zeros((batch.inputs.size(0), max_len)).long()
        tgt = tgt + self.sos_id

        for i in range(max_len):
            out = self.decode(
                memory = memory,
                tgt = tgt,
                doc = doc_hidden,
                src_mask = batch.input_mask,
                tgt_mask = None,
                doc_mask = batch.doc_mask,
                head_embeddings = head_embeddings,
                rel_embeddings = rel_embeddings,
                tail_embeddings = tail_embeddings,
                graph_mask = batch.graph_mask,
                history_text = batch.inputs,
                doc_text = batch.doc,
                head_text = batch.head_text,
                rel_text = batch.rel_text,
                tail_text = batch.tail_text
            )

            _, next_ids = torch.max(out, dim = -1)
            next_ids = next_ids[:, -1]
            outputs[:, i] = next_ids

            tgt = torch.cat([tgt, next_ids.unsqueeze(-1)], dim = -1)
            #print(tgt.size())
            if (tgt == self.eos_id).any(1).all().item():
                break
        
        return outputs.cpu().tolist()

    def purify(self, sents):
        res = []
        for sent in sents:
            new_sent = []
            for token in sent:
                if token == self.eos_id:
                    break
                elif token == self.pad_id:
                    continue
                new_sent.append(token)

            res.append(new_sent)
        return res




if __name__ == "__main__":
    #torch.manual_seed(3407)
    config = ModelConfig()
    config.read_json_config('./settings/model_config.json')
    ds = MyDataset("film", "film-new")
    #ds.make(src_dir = "./processed_data/KdConv/film", save_dir = './processed_data')
    ds.from_preprocessed('/data/yangpan/processed_data/film-new.pkl')
    md = Model(
        token_num = ds.num_token,
        ent_num = ds.graph_train_set.ent_num,
        rel_num = ds.graph_train_set.rel_num,
        pad_id = ds.pad_token_id,
        sos_id = ds.sos_token_id,
        eos_id = ds.eos_token_id,
        config = config
    )
    


    #md.resize_embedding(len(ds.tokenizer.get_vocab()))
    run_config = TrainConfig()
    run_config.read_json_config("./settings/train_config.json")
    loader = DataLoader(dataset = ds.train_set, batch_size = run_config["batch_size"], shuffle = True, num_workers = 2, collate_fn = ds.collate_fn)


    model = md.to(run_config["device"])
    model.load_graph_embedding_weights('/root/autodl-tmp/checkpoints')
    model.train_self(dataloader = loader, config = run_config)
