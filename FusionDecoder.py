import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionDecoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, nhead, drop_out, graph_attn_method, use_doc_attn, use_graph_attn) -> None:
        super().__init__()
        
        self.use_doc_attn = use_doc_attn
        self.use_graph_attn = use_graph_attn

        self.head_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.rel_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.tail_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.doc_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.nhead = nhead
        self.graph_attn_method = graph_attn_method

        self.W = nn.Linear(embed_dim, embed_dim, bias = False)

        self.fnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim)
        )
    def do_self_attn(self, reply, reply_mask):
        return self.self_attn.forward(
            query = reply,
            key = reply,
            value = reply,
            key_padding_mask = (reply_mask == 0) if reply_mask is not None else reply_mask,
            attn_mask = nn.Transformer.generate_square_subsequent_mask(reply.size(1)).to(reply.device)
        )[0]
    
    def do_doc_attn(self, reply, doc, doc_mask):
        res, doc_w = self.doc_attn.forward(
            query = reply,
            key = doc, 
            value = doc,
            key_padding_mask = (doc_mask == 0)
        )

        return res, doc_w
    
    def do_graph_attn(self, memory, head, rel, tail, graph_mask):
        _, wh = self.head_attn.forward(
            query = memory,
            key = head,
            value = head,
            key_padding_mask = (graph_mask == 0),
            average_attn_weights = False
        )

        _, wr = self.rel_attn.forward(
            query = memory,
            key = rel,
            value = rel,
            key_padding_mask = (graph_mask == 0),
            average_attn_weights = False
        )

        _, wt = self.tail_attn.forward(
            query = memory,
            key = tail,
            value = tail,
            key_padding_mask = (graph_mask == 0),
            average_attn_weights = False
        )

        graph_w = None
        if self.graph_attn_method == "mean-A":
            graph_w = (wh + wr + wt) / 3
        elif self.graph_attn_method == "mean-G":
            wh_ = torch.log(wh + 1e-12)
            wr_ = torch.log(wr + 1e-12)
            wt_ = torch.log(wt + 1e-12)

            graph_w = torch.exp((wh_ + wr_ + wt_) / 3)
            #print(graph_mask.view(graph_mask.size(0), 1, 1, graph_mask.size(1)).expand(-1, graph_w.size(1), graph_w.size(2), -1).shape)
            #print(graph_w.shape)
            graph_w = graph_w.masked_fill(graph_mask.view(graph_mask.size(0), 1, 1, graph_mask.size(1)).expand(-1, graph_w.size(1), graph_w.size(2), -1) == 0, float('-inf'))
            graph_w = graph_w.softmax(dim = -1)
            #graph_w = graph_w.softmax(graph_w.masked_fill(graph_mask.expand(graph_mask.size(0), graph_w.size(1), -1, -1), float('-inf')), dim = -1)
            #graph_w = graph_w.softmax(graph_w.masked_fill((graph_mask == 0).view(graph_mask.size(0), 1, 1, graph_mask.size(1)).expand(-1, graph_w.size(1), graph_w.size(2), -1), float('-inf')), dim = -1)
            #graph_w = graph_w.sum(dim = 1) / self.nhead
            #self.graph_w = F.softmax(graph_w.masked_fill((graph_mask == 0).view(graph_mask.size(0), 1, graph_mask.size(1)).expand(-1, graph_w.size(1), -1), dim = -1))
        
        elif self.graph_attn_method == "min":
            graph_w = torch.minimum(torch.minimum(wh, wr), wt)
        elif self.graph_attn_method == "raw":
            graph_w = wt        
        else:
            raise f"invalid attn method '{self.graph_attn_method}'"
        
        graph_w = graph_w.sum(dim = 1) / self.nhead

        return torch.matmul(graph_w, self.W(tail)), graph_w
    
    def do_cross_attn(self, memory, memory_mask, reply):
        res, history_w = self.cross_attn.forward(
            query = reply,
            key = memory,
            value = memory,
            key_padding_mask = memory_mask
        )
        return res, history_w
    
    def forward(self, memory, reply, doc, memory_mask, reply_mask, doc_mask, head, rel, tail, graph_mask):
        x = reply
        x = self.norm1(x + self.do_self_attn(x, reply_mask))
        h_attn, g_attn, d_attn = None, None, None
        #x = self.norm2(x + self.do_cross_attn(memory, memory_mask, reply))
        xm, h_attn = self.do_cross_attn(memory, memory_mask, reply)
        x = self.norm2(x + xm)

        #x = self.norm3(x + self.do_graph_attn(x, head, rel, tail, graph_mask) + self.do_doc_attn(x, doc, doc_mask))

        if self.use_graph_attn:
            xg, g_attn = self.do_graph_attn(x, head, rel, tail, graph_mask)
        
        if self.use_doc_attn:
            xd, d_attn = self.do_doc_attn(x, doc, doc_mask)

        if self.use_doc_attn and self.use_graph_attn:
            x = self.norm3(x + xg + xd)
        elif self.use_doc_attn:
            x = self.norm3(x + xd)
        elif self.use_graph_attn:
            x = self.norm3(x + xg)
        
        x = self.norm4(x + self.fnn(x))
        return x, h_attn, g_attn, d_attn


class FusionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nhead, drop_out, nlayer, graph_attn_method, use_doc_attn, use_graph_attn) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FusionDecoderLayer(embedding_dim, hidden_dim, nhead, drop_out, graph_attn_method, use_doc_attn = use_doc_attn, use_graph_attn = use_graph_attn) for i in range(nlayer)]
        )

    def forward(self, memory, reply, doc, memory_mask, reply_mask, doc_mask, head, rel, tail, graph_mask):
        x = reply
        for layer in self.layers:
            x, h_attn, g_attn, d_attn = layer(
                memory,
                x,
                doc,
                memory_mask,
                reply_mask,
                doc_mask,
                head,
                rel,
                tail,
                graph_mask
            )
        return x, h_attn, g_attn, d_attn