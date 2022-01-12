# Report for Assignment3 Unsupervised Learning

# Objective:
Based on the files provided, generate text (new ideas).

# Extraction of Data:
I have used only less text files for training purposes due to hardware limit on my computer.

From each text file, I've extracted only the 'RESULTS'

```
datafiles_path = "/content/drive/My Drive/ML_Assignment3/data"
datafiles_names = os.listdir(path=datafiles_path)

corpus_array = []

for i in range(len(datafiles_names)):
    path = os.path.join(datafiles_path,datafiles_names[i])
    try:
      if 'RESULTS' in open(path,'r').read():
         corpus_array.append(open(path).read()[open(path,'r').read().find('== RESULTS') + 10:open(path,'r').read().find('== ISSUES')].replace('\n',''))
    except:
        pass

# Minimizing corpus size:
corpus = corpus[:50]
```

# Model:
I have use a Bidirectional LSTM network for the purposes of generating new text.
```
# Creating model
model = tf.keras.models.Sequential()
model.add(Embedding(input_dim=total_words_count, output_dim=250, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(total_words_count))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```


# Post Fitting models
I trained the model for 20 epochs, with each epoch taking 20 minutes to complete.

# Results:
Used a function where the model will predict the occurring words after the given seed text.
```
def generate_text(seed_text,next_words,model):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predict_x=model1.predict(token_list)
        classes_x=np.argmax(predict_x,axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == classes_x:
                output_word = word
                break
        seed_text += ' ' + output_word
    return seed_text
```

## Few Outputs.
### 1.
```
generate_text('AI',10,model1)
```

AI the model the results the results the results of of


# Conclusion:
**The model is now able to generate new ideas ('RESULTS') with less grammatical errors.**