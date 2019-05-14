import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp


class BIMPM(nn.Module):

    def __init__(self, args, data):
        super(BIMPM, self).__init__()

        self.args = args
        self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_size
        self.l = self.args.num_perspective

        # self.c_embed_size = int(args['--char-embed-size'])
        # self.w_embed_size = int(args['--embed-size'])
        # self.l = int(args['--perspective'])
        # self.dropout_val = float(args['--dropout'])
        # self.bi_hidden = int(args['--bi-hidden-size'])
        # self.char_hidden = int(args['--char-hidden-size'])
        # self.rnn_type = args['--rnn-type']
        # self.char_layer_size = int(args['--char-lstm-layers'])
        # self.context_layer_size = int(args['--bilstm-layers'])
        # self.char_inp = vocab + 100
        # self.classes = class_size
        # self.char_use = args['--char']

        self.word_emb = nn.Embedding(num_embeddings=args.word_vocab_size,
                                        embedding_dim=args.word_dim)

        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        self.dropout = nn.Dropout(p=self.args.dropout)

        if self.args.use_char_emb:
            self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)

            self.char_lstm = nn.LSTM(input_size=self.args.char_dim,
                                     hidden_size=self.args.char_hidden_size,
                                     num_layers=1,
                                     bidirectional=False,
                                     batch_first=True)

        self.context_lstm = nn.LSTM(input_size=self.d,
                                    hidden_size=self.args.hidden_size,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        # ----- Matching Layer -----
        # w_i: (1, 1, hidden_size, l)
        for i in range(1, 9):
            setattr(self, f'w{i}', nn.Parameter(torch.rand(1, 1, self.args.hidden_size, self.l)))

        # ----- Aggregation Layer -----
        self.aggregation_lstm = nn.LSTM(input_size=self.l * 8,
                                        hidden_size=self.args.hidden_size,
                                        num_layers=1,
                                        bidirectional=True,
                                        batch_first=True)

        # ----- Prediction Layer -----
        self.pred_layer = nn.Sequential(
            nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.args.hidden_size * 2, self.args.class_size)
        )
        # self.ff1 = nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2)
        # self.ff2 = nn.Linear(self.args.hidden_size * 2, self.args.class_size)

        self.init_weights()
        self.reset_parameters()

    def reset_parameters(self):
        # ----- Word Representation Layer -----
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.word_emb.weight.data[0], -0.1, 0.1)

        if self.args.use_char_emb:
            nn.init.uniform_(self.char_emb.weight, -0.005, 0.005)
            # zero vectors for padding
            self.char_emb.weight.data[0].fill_(0)
            nn.init.kaiming_normal_(self.char_lstm.weight_ih_l0)
            nn.init.constant_(self.char_lstm.bias_ih_l0, val=0)
            nn.init.orthogonal_(self.char_lstm.weight_hh_l0)
            nn.init.constant_(self.char_lstm.bias_hh_l0, val=0)

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0)
        nn.init.constant_(self.context_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0)
        nn.init.constant_(self.context_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_hh_l0_reverse, val=0)

        # ----- Matching Layer -----
        for i in range(1, 9):
            w = getattr(self, f'w{i}')
            nn.init.kaiming_normal_(w)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_lstm.weight_ih_l0)
        nn.init.constant_(self.aggregation_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_lstm.weight_hh_l0)
        nn.init.constant_(self.aggregation_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_lstm.bias_hh_l0_reverse, val=0)

        # # ----- Prediction Layer ----
        # nn.init.uniform_(self.pred_la.weight, -0.005, 0.005)
        # nn.init.constant_(self.pred_fc1.bias, val=0)
        #
        # nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        # nn.init.constant_(self.pred_fc2.bias, val=0)

    def init_weights(self):
        for param in list(self.parameters()):
            nn.init.uniform_(param, -0.01, 0.01)

    def init_char_embed(self, c1, c2):
        c1_embed = self.char_emb(c1)
        char_p1 = self.char_lstm(c1_embed)
        c2_embed = self.char_emb(c2)
        char_p2 = self.char_lstm(c2_embed)

        # (batch, char_hidden_size * num_directions)
        return char_p1[0][:, -1], char_p2[0][:, -1]

    def cosine_similarity(self, prod, norm):
        # As set in PyTorch documentation
        eps = 1e-8
        norm = norm * (norm > eps).float() + eps * (norm <= eps).float()

        return prod / norm

    def full_matching(self, p1, p2, w_matrix):
        """
        :param p1: (batch, seq_len, hidden_size)
        :param p2: (batch, hidden_size)
        :param w_matrix: (1, 1, hidden_size, l)
        :return: (batch, seq_len, l)
        """
        # (batch, seq_len, hidden_size, l)

        p1 = torch.stack([p1] * self.l, dim=3)
        p1 = w_matrix * p1

        p1_seq_len = p1.size(1)
        p2 = torch.stack([p2] * p1_seq_len, dim=1)
        p2 = torch.stack([p2] * self.l, dim=3)
        p2 = w_matrix * p2

        result = F.cosine_similarity(p1, p2, dim=2)
        return result

    def maxpool_matching(self, p1, p2, w_matrix):
        """
        :param p1: (batch, seq_len, hidden_size)
        :param p2: (batch, seq_len, hidden_size)
        :param w_matrix: (1, 1, hidden_size, l)
        :return: (batch, seq, l)
        """
        # (batch, seq_len, hidden_size, l)
        p1 = torch.stack([p1] * self.l, dim=3)
        p1 = w_matrix * p1

        p2 = torch.stack([p2] * self.l, dim=3)
        p2 = w_matrix * p2

        # (batch, seq_len, 1, l)
        p1_norm = p1.norm(p=2, dim=2, keepdim=True)
        p2_norm = p2.norm(p=2, dim=2, keepdim=True)
        # (batch, l, seq_len1, seq_len2)
        full_mat = torch.matmul(p1.permute(0, 3, 1, 2), p2.permute(0, 3, 2, 1))
        deno_mat = torch.matmul(p1_norm.permute(0, 3, 1, 2), p2_norm.permute(0, 3, 2, 1))

        # (batch, seq1, l)
        result, _ = self.cosine_similarity(full_mat, deno_mat).max(dim=3)
        result = result.transpose(1, 2)
        return result

    def attentive_matching(self, p1, p2, w_matrix_att, w_matrix_max):
        """
        :param p1: (batch, seq_len1, hidden)
        :param p2: (batch, seq_len2, hidden)
        :param w_matrix_att: (1, 1, hidden, l)
        :param w_matrix_max: (1, 1, hidden, l)
        :return:
        """
        # Perform both attentive types of matching together
        p1_norm = p1.norm(p=2, dim=2, keepdim=True)
        p2_norm = p2.norm(p=2, dim=2, keepdim=True)

        # (batch, seq1, seq2)
        full_mat = torch.matmul(p1, p2.permute(0, 2, 1))
        deno_mat = torch.matmul(p1_norm, p2_norm.permute(0, 2, 1))
        alpha_mat = self.cosine_similarity(full_mat, deno_mat)

        # (batch, seq1)->(batch, seq1, hidden)
        _, max_index = torch.max(alpha_mat, dim=2)
        max_index = torch.stack([max_index] * self.args.hidden_size, dim=2)

        # (batch, seq1, hidden)
        h_mat = torch.bmm(alpha_mat, p2)
        alpha_mat = alpha_mat.sum(dim=2, keepdim=True)
        resultant = h_mat / alpha_mat

        # (batch, seq1, hidden, l)
        # (batch, seq1, hidden, l)
        # (batch, seq1, l)
        v1 = resultant.unsqueeze(-1) * w_matrix_att
        v2 = p1.unsqueeze(-1) * w_matrix_att
        result_match = F.cosine_similarity(v1, v2, dim=2)

        # (batch, seq1, l)
        out_mat = torch.gather(p2, 1, max_index)
        v1 = out_mat.unsqueeze(-1) * w_matrix_max
        v2 = p1.unsqueeze(-1) * w_matrix_max
        result_max = F.cosine_similarity(v1, v2, dim=2)

        return result_match, result_max

    def forward(self, **kwargs):

        p1_input = self.word_emb(kwargs['p'])
        p2_input = self.word_emb(kwargs['h'])

        if self.args.use_char_emb:
            char_p1, char_p2 = self.init_char_embed(kwargs['char_p'], kwargs['char_h'])
            dim1, dim2 = kwargs['p'].size()
            char_p1 = char_p1.view(dim1, dim2, -1)
            dim1, dim2 = kwargs['h'].size()
            char_p2 = char_p2.view(dim1, dim2, -1)
            p1_input = torch.cat((p1_input, char_p1), -1)
            p2_input = torch.cat((p2_input, char_p2), -1)

            context1_full, (context1_lh, _) = self.context_lstm(p1_input)
            context2_full, (context2_lh, _) = self.context_lstm(p2_input)

        else:
            context1_full, (context1_lh, _) = self.context_lstm(p1_input)
            context2_full, (context2_lh, _) = self.context_lstm(p2_input)

        # (batch, seq_len, hidden_size)
        context1_forw, context1_back = torch.split(context1_full, self.args.hidden_size, -1)
        # (batch, hidden_size)
        # context1_lh_forw, context1_lh_back = context1_lh[0], context1_lh[1]
        context1_lh_forw, context1_lh_back = context1_forw[:, -1], context1_back[:, -1]

        context2_forw, context2_back = torch.split(context2_full, self.args.hidden_size, -1)
        context2_lh_forw, context2_lh_back = context2_forw[:, -1], context2_back[:, -1]
        # print(context2_lh_forw.shape, context2_lh_back.shape)
        # 4 tensors from forward and backward matching (full matching)
        match_p1_forw = self.full_matching(context1_forw, context2_lh_forw, self.w1)
        match_p1_back = self.full_matching(context1_back, context2_lh_back, self.w2)
        match_p2_forw = self.full_matching(context2_forw, context1_lh_forw, self.w1)
        match_p2_back = self.full_matching(context2_back, context1_lh_back, self.w2)

        # 4 tensors from forward and backward matching (max-pooling matching)
        maxm_p1_forw = self.maxpool_matching(context1_forw, context2_forw, self.w3)
        maxm_p1_back = self.maxpool_matching(context1_back, context2_back, self.w4)
        maxm_p2_forw = self.maxpool_matching(context2_forw, context1_forw, self.w3)
        maxm_p2_back = self.maxpool_matching(context2_back, context1_back, self.w4)

        # 8 tensors from the forward and backward attentive matching and attentive max
        att_p1_forw, attm_p1_forw = self.attentive_matching(context1_forw, context2_forw, self.w5, self.w7)
        att_p1_back, attm_p1_back = self.attentive_matching(context1_back, context2_back, self.w6, self.w8)
        att_p2_forw, attm_p2_forw = self.attentive_matching(context2_forw, context1_forw, self.w5, self.w7)
        att_p2_back, attm_p2_back = self.attentive_matching(context2_back, context1_back, self.w6, self.w8)

        aggr_p1 = torch.cat([match_p1_forw, match_p1_back, maxm_p1_forw, maxm_p1_back,
                             att_p1_forw, att_p1_back, attm_p1_forw, attm_p1_back], dim=2)

        aggr_p2 = torch.cat([match_p2_forw, match_p2_back, maxm_p2_forw, maxm_p2_back,
                             att_p2_forw, att_p2_back, attm_p2_forw, attm_p2_back], dim=2)

        aggr_p1 = self.dropout(aggr_p1)
        aggr_p2 = self.dropout(aggr_p2)

        # ----- Aggregation Layer -----

        # _, (p1_output, _) = self.aggregation_lstm(aggr_p1)
        # _, (p2_output, _) = self.aggregation_lstm(aggr_p2)
        #
        # output = torch.cat(
        #     [p1_output.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)
        #                     torch.cat([p2_output[0, :, :], p2_output[1, :, :]], dim=-1)], dim=-1)

        agg_p_last, _ = self.aggregation_lstm(aggr_p1)
        agg_h_last, _ = self.aggregation_lstm(aggr_p2)

        output = torch.cat([agg_p_last[:, -1], agg_h_last[:, -1]], dim=1)
        output = self.dropout(output)
        print("hhh")
        # ----- Prediction Layer -----
        # output = torch.tanh(self.ff1(output))
        # output = self.dropout(output)
        # output = self.ff2(output)
        output = self.pred_layer(output)
        return output
