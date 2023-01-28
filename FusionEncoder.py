import torch.nn as nn
import torch 
import torch.nn.functional as F


class EncoderFusionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, nhead, drop_out, graph_attn_method, use_doc_attn, use_graph_attn) -> None:
        super().__init__()

        self.use_doc_attn = use_doc_attn
        self.use_graph_attn = use_graph_attn

        self.head_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.rel_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.tail_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout=drop_out,batch_first=True)
        self.doc_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout = drop_out, batch_first = True)
        self.doc_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, dropout = drop_out, batch_first = True)

        self.nhead = nhead
        self.graph_attn_method = graph_attn_method

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.norma = nn.LayerNorm(embed_dim)
        self.normb = nn.LayerNorm(embed_dim)

        self.W = nn.Linear(embed_dim, embed_dim, bias = False)

        self.fnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.fnn2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim)            
        )
    def do_self_attn(self, src, src_mask):
        return self.self_attn.forward(
            query = src,
            key = src,
            value = src,
            key_padding_mask = (src_mask == 0)
        )[0]
    
    def do_doc_self_attn(self, doc, doc_mask):
        return self.doc_self_attn.forward(
            query = doc,
            key = doc,
            value = doc,
            key_padding_mask = (doc_mask == 0)
        )[0]
    
    def do_doc_attn(self, doc, src, doc_mask):
        return self.doc_attn.forward(
            query = src,
            key = doc,
            value = doc,
            key_padding_mask = (doc_mask == 0)
        )[0]

    def do_graph_attn(self, src, head, rel, tail, graph_mask):

        _, wh = self.head_attn.forward(
            query = src,
            key = head,
            value = head,
            key_padding_mask = (graph_mask == 0),
            average_attn_weights = False
        )

        _, wr = self.rel_attn.forward(
            query = src,
            key = rel,
            value = rel,
            key_padding_mask = (graph_mask == 0),
            average_attn_weights = False
        )

        _, wt = self.tail_attn.forward(
            query = src,
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
            #graph_w = graph_w.softmax(graph_w.masked_fill((graph_mask == 0).view(graph_mask.size(0), 1, 1, graph_mask.size(1)).expand(graph_mask.size(0), graph_w.size(1), graph_w.size(2), -1), float('-inf')), dim = -1)
            graph_w = graph_w.masked_fill(graph_mask.view(graph_mask.size(0), 1, 1, graph_mask.size(1)).expand(-1, graph_w.size(1), graph_w.size(2), -1) == 0, float('-inf'))
            graph_w = graph_w.softmax(dim = -1)
            #graph_w = graph_w.sum(dim = 1) / self.nhead
            #self.graph_w = F.softmax(graph_w.masked_fill((graph_mask == 0).expand(graph_mask.size(0), graph_w.size(1), -1), dim = -1))
            #self.graph_w = F.softmax(graph_w.masked_fill((graph_mask == 0).view(graph_mask.size(0), 1, graph_mask.size(1)).expand(-1, graph_w.size(1), -1), dim = -1))
        elif self.graph_attn_method == "min":
            graph_w = torch.minimum(torch.minimum(wh, wr), wt)
        elif self.graph_attn_method == "raw":
            graph_w = wt
        else:
            raise f"invalid attn method '{self.graph_attn_method}'"
        
        graph_w = graph_w.sum(dim = 1) / self.nhead

        return torch.matmul(graph_w, self.W(tail))

    def forward(self, src, doc, src_mask, doc_mask, head, rel, tail, graph_mask):
        x = src
        y = doc
        y = self.norma(y + self.do_doc_self_attn(y, doc_mask))
        #print("doc1", torch.any(torch.isnan(y)))
        x = self.norm1(x + self.do_self_attn(x, src_mask))
        #print("input1", torch.any(torch.isnan(x)))
        
        if self.use_doc_attn and self.use_graph_attn:
            x = self.norm2(x + self.do_graph_attn(x, head, rel, tail, graph_mask) + self.do_doc_attn(y, x, doc_mask))
        elif self.use_doc_attn:
            x = self.norm2(x + self.do_doc_attn(y, x, doc_mask))
        elif self.use_graph_attn:
            x = self.norm2(x + self.do_graph_attn(x, head, rel, tail, graph_mask))
        
        #print("input2", torch.any(torch.isnan(x)))
        x = self.norm3(x + self.fnn(x))
        #print("input3", torch.any(torch.isnan(x)))
        y = self.normb(y + self.fnn2(y))
        #print("input4", torch.any(torch.isnan(y)))

        return x, y


class FusionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nhead, drop_out, nlayer, graph_attn_method, use_doc_attn, use_graph_attn) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderFusionLayer(embedding_dim, hidden_dim, nhead, drop_out, graph_attn_method, use_doc_attn = use_doc_attn, use_graph_attn = use_graph_attn) for i in range(nlayer)]
        )

    def forward(self, src, doc, src_mask, doc_mask, head, rel, tail, graph_mask):
        #print(graph_mask.shape)
        x = src
        y = doc
        for layer in self.layers:
            x, y = layer(
                x,
                y,
                src_mask,
                doc_mask,
                head,
                rel,
                tail,
                graph_mask
            )
        return x, y
    