import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.autograd as autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = self.generateEmbedding()
        self.lstm = self.generateLSTM()
        self.linear_layer = nn.Linear(hidden_dim*2, hidden_dim)
    
    def generateEmbedding(self):
        return nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
    
    def generateLSTM(self):
        return nn.LSTM(self.embed_dim, self.hidden_dim, bidirectional=True)
    
    def forward(self, inputs, hidden=None):
        #inputs:(seq_len, batch_size)
        #hidden:(2, batch_size, hidden_dim)
        batch_size = inputs.size(1)
        if hidden is None:
            hidden = (autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)).to(device),
                      autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)).to(device))
        embeds = self.embedding((inputs))
        #embeds:(seq_len, batch_size, embed_dim)
        outputs, hidden = self.lstm(embeds, hidden)
        #outputs:(seq_len, batch_size, hidden_dim*num_directions)
        #hidden:(2, batch_size, hidden_dim)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, ht, hs):
        #ht:(1, batch_size, hidden_dim)
        #hs:(seq_len, batch_size, hidden_dim)
        ht = ht.permute(1, 0, 2)
        #ht:(batch_size, 1, hidden_dim)
        hs = hs.permute(1, 0, 2)
        #hs:(batch_size, seq_len, hidden_dim)
        Whs = self.linear_layer(hs)
        #Whs:(batch_size, seq_len, hidden_dim)
        score = torch.bmm(ht, Whs.permute(0, 2, 1))
        #score:(batch_size, 1, seq_len)
        weights = F.softmax(score, dim=1)
        return weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = self.generateEmbedding()
        self.lstm = self.generateLSTM()
        self.attention = Attention(hidden_dim)
        self.linear_layer1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, vocab_size)

    def generateEmbedding(self):
        return nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
    
    def generateLSTM(self):
        return nn.LSTM(self.embed_dim, self.hidden_dim)
    
    def forward(self, inputs, encoder_outputs, hidden=None):
        #inputs:(batch_size)
        #hidden:(1, batch_size, hidden_dim)
        #encoder_outputs:(seq_len, batch_size, hidden_dim)
        inputs = inputs.unsqueeze(0)
        #inputs:(1, batch_size)
        embeds = self.embedding((inputs))
        #embeds:(1, batch_size, embed_dim)
        lstm_output, hidden = self.lstm(embeds, hidden)
        #lstm_output:(1, batch_size, hidden_dim)
        #hidden:(1, batch_size, hidden_dim) 
        weights = self.attention(lstm_output, encoder_outputs)
        #weights:(batch_size, 1, seq_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs:(batch_size, seq_size, hidden_dim)
        context = weights.bmm(encoder_outputs).permute(1, 0, 2)
        #context:(1, batch_size, hidden_dim)
        context = torch.cat((context, lstm_output), 2)
        #context:(1, batch_size, hidden_dim*2)
        context = self.linear_layer1(context)
        #context:(1, batch_size, hidden_dim)
        ht = torch.tanh(context)
        #ht:(1, batch_size, hidden_dim)
        outputs = self.linear_layer2(ht)
        #outputs:(1, batch_size, vocab_size)
        outputs = F.softmax(outputs, dim=2)
        outputs = outputs.squeeze(0)
        return outputs, hidden, weights


class Net(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src_seqs, tag_seqs):
        #src_seqs:(seq_len, batch size)
        #tag_seqs:(seq_len, batch size)
        batch_size = src_seqs.size(1)
        max_len = tag_seqs.size(0)
        tag_vocab_size = self.decoder.vocab_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, tag_vocab_size).to(self.device)
        #outputs:(max_len, batch_size, vocab_size)
        encoder_outputs, hidden = self.encoder(src_seqs)
        #encoder_outputs:(seq_len, batch_size, hidden_dim*2)
        #hidden:(2, batch_size, hidden_dim)
        hidden_dim = hidden[0].size(2)
        encoder_outputs = encoder_outputs[:,:,:hidden_dim] + encoder_outputs[:,:,hidden_dim:]
        hidden = ((hidden[0][:1,:,:]+hidden[0][1:,:,:]),
                  (hidden[1][:1,:,:]+hidden[1][1:,:,:]))
        #first input to the decoder is <EOS>
        output = tag_seqs[0, :]
        #output:(batch_size)
        
        for t in range(1, max_len): #skip <EOS>
            output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
            #output:(batch_size, vocab_size)
            #hidden:(1, batch_size, hidden_dim)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            output = (tag_seqs[t] if teacher_force else output.max(1)[1])
            #outputs:(max_len, batch_size, vocab_size)
        return outputs

    def predict(self, src_seqs, start_tag, max_tag_len=100, topk=2):
        #src_seqs:(src_len, 1)
        batch_size = src_seqs.size(1)
        outputs = torch.zeros(max_tag_len, batch_size).to(self.device)
        #outputs:(max_tag_len, 1)
        encoder_outputs, hidden = self.encoder(src_seqs)
        #encoder_outputs:(seq_len, 1, hidden_dim*2)
        #hidden:(2, 1, hidden_dim)
        hidden_dim = hidden[0].size(2)
        encoder_outputs = encoder_outputs[:,:,:hidden_dim] + encoder_outputs[:,:,hidden_dim:]
        hidden = ((hidden[0][:1,:,:]+hidden[0][1:,:,:]),
                  (hidden[1][:1,:,:]+hidden[1][1:,:,:]))
        output = torch.LongTensor([start_tag]).to(self.device)
        
        output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
        #output:(1, vocab_size)
        #hidden:(1, 1, hidden_dim)
        index = self.beam_search(output, encoder_outputs, hidden, max_tag_len, topk)
        outputs[1][0] = index  
        output, hidden, _ = self.decoder(torch.LongTensor([index]).to(self.device), encoder_outputs, hidden)
        output = output.max(1)[1]

        for t in range(2, max_tag_len):
            output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
            #output:(1, vocab_size)
            #hidden:(1, 1, hidden_dim)
            outputs[t][0] = output.max(1)[1]
            output = output.max(1)[1]
        return outputs
    
    def beam_search(self, output, encoder_outputs, hidden, max_tag_len, topk):
        topk_vocab = output[0].topk(topk)
        max_index = None
        max_prob = 0.0
        for item in topk_vocab:
            prob = item[0].item()
            index = item[1]
            output, hidden, _ = self.decoder(torch.LongTensor([index]).to(self.device), encoder_outputs, hidden)
            output = output.max(1)[1]
            for t in range(3, max_tag_len):
                output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
                prob += output.max(1)[0].item()
                output = output.max(1)[1]
            if prob>max_prob:
                max_prob = prob
                max_index = index
        return max_index
            
        
        
        
        
        
        
        
        
#        #src_seqs:(src_len, batch_size)
#        #start_tag
#        
#        #tensor to store decoder outputs
#        outputs = torch.zeros(max_tag_len).to(self.device)
#        #outputs:(max_tag_len)
#        encoder_outputs, hidden = self.encoder(src_seqs)
#        #encoder_outputs:(src_seq_len, 1, hidden_dim*2)
#        #hidden:(2, 1, hidden_dim)
#        hidden_dim = hidden[0].size(2)
#        encoder_outputs = encoder_outputs[:,:,:hidden_dim] + encoder_outputs[:,:,hidden_dim:]
#        hidden = ((hidden[0][:1,:,:]+hidden[0][1:,:,:]),
#                  (hidden[1][:1,:,:]+hidden[1][1:,:,:]))
        
#        output = start_tag
#        outputs[0] = start_tag
#        output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
#        #output:(1, vocab_size)
#        #hidden:(1, batch_size, hidden_dim)  
#        output = output.reshape(-1)
#        topk_tag = torch.topk(output, k=topk)
#        (max_i, max_output, max_hidden) = (0, None, None)
#        for i in range(topk):
#            i_tag = topk_tag[i]
#            i_prob, output, hidden = self.beam_search(i_tag, max_tag_len, encoder_outputs, hidden)
#            if i == 0:
#                max_prob = i_prob
#                max_i, max_output, max_hidden = i_tag, output, hidden
#            else:
#                if i_prob > max_prob:
#                    max_prob = i_prob
#                    max_i, max_output, max_hidden = i_tag, output, hidden
#        outputs[1] = max_i 
#        output = max_output
#        hidden = max_hidden
#                          
#        for t in range(2, max_tag_len):
#            output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
#            #output:(batch_size, vocab_size)
#            #hidden:(1, batch_size, hidden_dim)
#            output.reshape(-1)
#            outputs[t] = output
#            output = output.max()[1]
#        return outputs
#
#    def beam_search(self, start_tag, max_tag_len, encoder_outputs, hidden):
#        prob = start_tag[0]
#        output = start_tag[1]
#        for t in range(2, max_tag_len):
#            output, hidden, _ = self.decoder(output, encoder_outputs, hidden)
#            #output:(batch_size, vocab_size)
#            #hidden:(1, batch_size, hidden_dim)
#            output = output.max(1)[1]
#            prob += output.max(1)[0]
#        return prob, output, hidden
        
        
        
        
        
        
        
        

