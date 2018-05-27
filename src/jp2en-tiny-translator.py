# coding: utf-8
from io import open
import unicodedata
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
font = {"family":"Osaka"}
plt.rc('font', **font)

import sys,os
sys.path.append(os.pardir)
from utilities.NLPtools import *

use_cuda = torch.cuda.is_available()

# ---- Pre-processing ----
# Parameter Initialization
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

teacher_forcing_ratio = 0.5
hidden_size = 256

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# ---- Define Functions and Classes ----
def unicodeToAscii(s):
    """
    Extract unicode data
    :param s: sentence in unicode format
    :return: sentence in ascii format
    >>> unicode_to_ascii('a&bあ')
    >>> 'ab'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Normalize string by inserting space before .!?
    :param s: original string
    :return: normalized string
    """
    s = s.lower().strip()
    #s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence, lang_name):
        """
        After splitting to word chunk, update word2index, word2count, index2word list
        :param sentence: input sentence
        :lang_name: name of language
        :return: self instance with updated property
        """
        if lang_name in ['eng', 'fra']:
            for word in sentence.split(' '):
                self.addWord(word)
        elif lang_name == 'jpn':
            wakachi = mcb_tokenize(sentence)
            for word in wakachi:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    """
    Open and read file to create sentence pair
    :param lang1: First language name
    :param lang2: Second language name
    :param reverse: swap lang1 and lang2
    :return: input language, output language and sentence pair
    >>> input_lang.name
    >>>  'fra'
    >>> output_lang.name
    >>>  'eng'
    >>> pairs
    >>> [['va !', 'go .'],
    >>>  ['cours !', 'run !'],
    >>>  ['courez !', 'run !'],..
    """
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/machine_translation/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    #pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    """
    If sentences length are less than 10 and English sentence starts with
    prefix like 'i am', 'he is', return True
    :param p: sentence pair
    :return: True if condition is satisfied
    """
    # return len(p[0].split(' ')) < MAX_LENGTH and \
    #     len(p[1].split(' ')) < MAX_LENGTH and \
    #     p[1].startswith(eng_prefixes)
    return len(mcb_tokenize(p[0])) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    """
    Extract pairs to fulfill condition in filterPair
    It is necessary to reduce time and improve accuracy for efficiency
    :param pairs: sequence pair list
    :return: filtered sequence pair list
    """
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    """
    After reading sentence pair from file, filter them out, and append to word2index, word2count, index2word list
    :param lang1: First language
    :param lang2: Second language
    :param reverse: option to swapt lang1 and lang2
    :return: input_lang, output_lang, pairs
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0],input_lang.name)
        output_lang.addSentence(pair[1],output_lang.name)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    #fra 4589
    #eng 2988
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    """
     A GRU based seq2seq network that outputs embedded value for every word
     from the input sentence
     For every input word the encoder outputs a vector and a hidden state,
     and uses the hidden state for the next input word.
     input->embedding->embedded->gru-> output
     prev_hiddedn-------^           -> hidden
    """
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    """
    A simple decoder to takes the encoder output vector(s) and outputs
    a sequence of words to create the translation.
    input->embedding->relu->gru->out->softmax->output
    prev_hiddedn-------^       --------------->hidden
    """
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    """
    Decoder with Attention which allows the decoder network to “focus” on a different part
    of the encoder’s outputs for every step of the decoder’s own outputs.
    """
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            #self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        #output = F.log_softmax(self.out(output[0]), dim=1)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def indexes_from_sentence(lang, sentence):
    """
    Extract indexes from a sentence (sentence->word->index)
    :param lang: language
    :param sentence: input sentence
    :return: list of indexes
    >>> lang.name
    >>> 'fra'
    >>> sentence
    >>> 'mes parents m adorent .'
    >>>  [652, 2632, 477, 3003, 5, 1]
    """

    if lang.name in ['eng','fra']:
        return [lang.word2index[word] for word in sentence.split(' ')]
    if lang.name == 'jpn':
        return [lang.word2index[word] for word in mcb_tokenize(sentence)]

def variable_from_sentence(lang, sentence):
    """
    Convert every word into index and put them together with EOS
    After that, change the indexes to Variable format
    Size of result is determined based on sentence length
    (No need to be fixed always)
    :param lang: language
    :param sentence: input sentence
    :return: Variables of a sentence
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(pair):
    """
    Generate sentence index Variables for input and output
    Then create a list to store them as pair
    :param pair: sentence pair
    :return: Variable sentence index pair
    """
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


def as_minutes(s):
    """
    Convert second to minutes and second
    :param s: second
    :return: minutes and second string
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    """
    Calculate time since 'since'
    :param since: started time
    :param percent:
    :return:
    """
    now = time.time()
    s = now - since  # erapsed time [s]
    es = s / (percent)  # erapsed time [ratio]
    rs = es - s  # remaining time
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    """
    Prepare plot data
    :param points: plot data point
    :return: None
    """
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
    Perform training
    :param input_variable:
    :param target_variable:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion:
    :param max_length:
    :return: Loss data
    """
    # Initialization
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Encoding
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    # Decoding
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # use_teacher_forcing is activated randomly (once in every twice)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # Calculate loss function
            loss += criterion(decoder_output, target_variable[di])

            # Teacher forcing
            decoder_input = target_variable[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # Calculate input to decoder
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            # Calculate loss function
            loss += criterion(decoder_output, target_variable[di])

            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """
    Perform training in each iteration
    :param encoder: encoder instance
    :param decoder: decoder instance
    :param n_iters: number of iteration
    :param print_every: timing to print
    :param plot_every: timing to plot
    :param learning_rate: learning rate
    :return: None
    """
    # Initiazliaation
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Define opimizer
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Randomly pick up variable pairs
    training_pairs = [variables_from_pair(random.choice(pairs)) for i in range(n_iters)]

    # Define criteria to compute loss function
    criterion = nn.NLLLoss()

    # In every iteration, calculate loss using train() function
    # The loss is used for print and plot
    for iter in range(1, n_iters + 1):

        # Extract a variable pair for training
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Compute loss
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Print loss data
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # 11m 20s (- 158m 47s) (5000 6%) 2.9332
            # 23m 19s (- 151m 34s) (10000 13%) 2.3810
            # ..
            # 174m 24s (- 0m 0s) (75000 100%) 0.6343

        # Plot loss data
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # Prepare data for plot (Later the data is shown in show_attention())
    show_plot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
    Perform encoding and decoding for evaluation
    :param encoder: encoder instance
    :param decoder: decoder instance
    :param sentence: input sentence
    :param max_length: maximum length of encoder output
    :return: decoded words and attention
    """
    # Prepare input variable
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    # Prepare variable for encoder
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # Perform encoding
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    # Prepare variable for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Perform decoding
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=10):
    """
    Randomly pickup sentence pair
    :param encoder: encoder instance
    :param decoder: decoder instance
    :param n: numbers of evaluation
    :return: None
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        # > vous etes aveuglee par l amour .
        # = you are blinded by love .
        # < you are blinded by love . <EOS>

def show_attention(input_sentence, output_words, attentions):
    """
    Show attention graphically
    :param input_sentence:
    :param output_words:
    :param attentions:
    :return:
    """

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_xticklabels([''] + mcb_tokenize(input_sentence) + ['<EOS>'], rotation=90)

    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(input_sentence):
    """
    Perform evaluaton and plot figures (loss, attention) for sample input sentence
    :param input_sentence: input sentence
    :return: None
    """
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)




# ---- Encoder and Decoder Generation ----
# Generate language pairs and print one pair randomly selected
input_lang, output_lang, pairs = prepareData('eng', 'jpn', True)
print(random.choice(pairs))
# ['nous sommes tous fous .', 'we re all crazy .']

# Generate encoder instance and attention decoder instance
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,  1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

# ---- Training ----
train_iters(encoder1, attn_decoder1, 75000, print_every=5000)


# ---- Evaluation ----
evaluate_randomly(encoder1, attn_decoder1)


output_words, attentions = evaluate(encoder1, attn_decoder1, "彼は生まれつき寛大な人だ。")
plt.matshow(attentions.numpy())


evaluate_and_show_attention("彼は見聞の広い人だ。")
# input = 彼は見聞の広い人だ。
# output = he is a well informed person. <EOS>

evaluate_and_show_attention("私はちょうど散歩に出かけるところです。")
# input = 私はちょうど散歩に出かけるところです。
# output = i am just going for a walk. <EOS>

evaluate_and_show_attention("彼女は彼より頭がいい。")
# input = 彼女は彼より頭がいい。
# output = she is smarter than he is. <EOS>
