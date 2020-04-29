#!pip3 install tensorflow==1.0.0 ##This version suits

##Importing the libraries

import numpy as np
import tensorflow as tf
import re
import pandas as pd
import time


###### Importing Dataset and data preprocessing ######

lines =open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversation =open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


#Create a dictionary to map each line with id
id2line={} #mapping id to lines

for line in lines:
    _line=line.split(" +++$+++ ")
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
        
#Create a list of all conversation

conversations_ids=[]
for converse in conversation[:-1]: ## the last row is an empty row. exclude that and take the entire dataset
    _conversation = converse.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","") ##[1:-1] Excludes the first and the last index
    ## and replace the apostrophes with nothing
    ## and replace the spaces with nothing
    conversations_ids.append(_conversation.split(','))
    
    
#Getting separately the questions and the answers
    
questions =[]
answers =[]

for converse in conversations_ids:
    for i in range(len(converse)-1):
        questions.append(id2line[converse[i]])   ## questions text
        answers.append(id2line[converse[i+1]])   ## answers text

#Doing the first cleaning of the text


def clean_text(text):
    text = text.lower() #lowercase the text
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"there's","there is",text)
    text = re.sub(r"didn't","did not",text)
    text = re.sub(r"don't","do not",text)
    text = re.sub(r"it's","it is",text)
    text = re.sub(r"doesn't","does not",text)
    text = re.sub(r"haven't","have not",text)
    text = re.sub(r"c'mon","come on",text)
    text = re.sub(r"let's","let us",text)
    text = re.sub(r"&","and",text)
    text = re.sub(r"[-()\"!#@;%$:<>{}+~*|.?,&]","",text)  ## remove all special characters
    return text
    
#Cleaning the questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))

#Cleaning the answers    
clean_answers=[]
for ans in answers:
    clean_answers.append(clean_text(ans))


#Creating a dictionary that maps words to their number of occurences / frequency

word2count={}
for ques in clean_questions:
    for word in ques.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


word2count={}
for ans in clean_answers:
    for word in ans.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
            
            
#Creating a dictionary to map question and answers to a unique integer // Filtering
            
threshold= 20
questionword2int={}
word_number=0

for word, count in word2count.items():
    if count>=threshold:
        questionword2int[word]= word_number
        word_number+=1
        
answerword2int={}
word_number=0

for word, count in word2count.items():
    if count>=threshold:
        answerword2int[word]= word_number
        word_number+=1           
            

#Adding last tokens to the dictionaries

tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
           
for token in tokens:
    questionword2int[token]=len(questionword2int) + 1
    answerword2int[token]=len(answerword2int) + 1

      
#Creating the inverse dictionary of the answerword2int dictionary
    
answerint2word={w_i:w for w, w_i in answerword2int.items()}            
            
            
# Adding the <EOS> token to the end of every answer

for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'          
            

#Translating all the questions and answers into integers
#and replacing all the words below threshold frequency by <OUT>

questions_to_int =[] 

for ques in clean_questions:
    ints=[]
    for word in ques.split():
        if word not in questionword2int:
            ints.append(questionword2int['<OUT>'])
        else:
            ints.append(questionword2int[word])
    questions_to_int.append(ints)
            
            
            
answers_to_int =[] 

for ans in clean_answers:
    ints=[]
    for word in ans.split():
        if word not in answerword2int:
            ints.append(answerword2int['<OUT>'])
        else:
            ints.append(answerword2int[word])
    answers_to_int.append(ints)           
            

# Sorting the questions and answers by the length of the question (Reduces amount of PADDING and increases training speed)

sorted_clean_questions=[]
sorted_clean_answers=[]

for length in range(1, 25 + 1): # only keep those sentences whose lenght is <=25
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
          
            

###### SEQ2SEQ TRAINING  #########          
            
#Creating placeholders for the inputs and output

def model_inputs():
    inputs=tf.placeholder(tf.int32, [None,None], name = 'input') ## 2 dimensional input vector
    targets=tf.placeholder(tf.int32, [None,None], name = 'target') 
    lr=tf.placeholder(tf.float32, name = 'learning_rate')      
    keep_prob=tf.placeholder(tf.float32, name = 'keep_prob') ## Used for dropout
    return inputs, targets, lr, keep_prob

    
#Preprocess the targets
    
def preprocess_targets(targets,word2int,batch_size):
    left_side=tf.fill([batch_size,1],word2int['<SOS>']) ##all data into batchsize with SOS at front of each data
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1]) ## all the rows in batch except the last column(ALL ANSWERS EXCEPT THE LAST)
    preprocessed_targets = tf.concat([left_side,right_side],axis=1)##horizontal concatenation
    return preprocessed_targets

#Creating the encoder RNN Layer
    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequence_length =sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32) 
    ## _, is for getting only the second element
    return encoder_state
    

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
    
    


 ########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########
 
 
 
# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerword2int),
                                                       len(questionword2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionword2int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionword2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerword2int))
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
    
    
########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########
 
 
 
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Exit Please':
        break
    question = convert_string2int(question, questionword2int)
    question = question + [questionword2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answerint2word[i] == 'i':
            token = ' I'
        elif answerint2word[i] == '<EOS>':
            token = '.'
        elif answerint2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answerint2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    