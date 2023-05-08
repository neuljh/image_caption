import sys

import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import cv2
from keras.layers import Lambda
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Flatten, RepeatVector, TimeDistributed, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

import random

# Define the maximum length of the caption
max_len = 30
# Define the number of rows and columns in the grid
rows = 5
cols = 1
# Define the figure size
fig_size = (10, 10)
# define some num
EPOCHS = 10
BATCH_SIZE = 8
# data size
train_percent = 0.03
val_percent = 0.2
score_percent = 0.01
# 设置路径
images_dir = './data/coco/'
annotations_dir = './data/coco/annotations_trainval2017/annotations/'
train_image_dir = os.path.join(images_dir, 'train2017')
val_image_dir = os.path.join(images_dir, 'val2017')
train_caption_file = os.path.join(annotations_dir,  'captions_train2017.json')
val_caption_file = os.path.join(annotations_dir,  'captions_val2017.json')
print("train_caption_file: "+train_caption_file)
print("val_caption_file: "+train_caption_file)

def caculate_evaluation_indicator(model, val_set, word_index, reverse_word_index):
    # val_set_percent = val_set[:int(score_percent*len(val_set))]
    val_set_percent = random.sample(val_set, int(score_percent * len(val_set)))
    reference_captions = []
    predicted_captions = []
    for data in tqdm(val_set_percent, file=sys.stdout):
        reference_captions.append(data['caption'])
        predicted_captions.append(generate_caption(model, data['image'], word_index, reverse_word_index, True))
    print(reference_captions)
    print(predicted_captions)

    reference_corpus = {}
    predicted_corpus = {}
    for i, sentence in tqdm(enumerate(reference_captions), file=sys.stdout):
        reference_corpus[i] = [sentence]
    print(reference_corpus)
    for i, sentence in tqdm(enumerate(predicted_captions), file=sys.stdout):
        predicted_corpus[i] = [sentence]
    print(predicted_corpus)

    bleu_scorer = Bleu(n=4)  # n代表Bleu的n-gram，这里设置为4，即计算BLEU-1到BLEU-4
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    meteor_scorer = Meteor()
    spice_scorer = Spice()

    bleu_score, _ = bleu_scorer.compute_score(reference_corpus, predicted_corpus)
    # 计算ROUGE分数
    rouge_scores = rouge_scorer.compute_score(reference_corpus, predicted_corpus)
    cider_scores = cider_scorer.compute_score(reference_corpus, predicted_corpus)
    meteor_scores = meteor_scorer.compute_score(reference_corpus, predicted_corpus)
    # # 计算SPICE指标
    # spice_scores, _ = spice_scorer.compute_score(reference_corpus, predicted_corpus)

    print(bleu_score)
    print(rouge_scores)
    print(cider_scores)
    print(meteor_scores)
    # print(spice_scores)

    BLEU_1 = bleu_score[0]
    BLEU_2 = bleu_score[1]
    BLEU_3 = bleu_score[2]
    BLEU_4 = bleu_score[3]
    ROUGE = rouge_scores[0]
    CIDEr = cider_scores[0]
    METEOR = meteor_scores[0]
    # SPICE = spice_scores['Spice']

    return BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, CIDEr, METEOR

def show_image(path):
    img = cv2.imread(path)
    if img is not None:
        cv2.namedWindow(path, 0)
        cv2.resizeWindow(path, 500, 500)
        cv2.imshow(path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Could not read image from {path}")

# Generate captions for a sample image
def generate_caption(model, image_file_path_or_image_data, word_index, reverse_word_index, is_data, max_len=30):
    if is_data is False:
        # open the image
        image = Image.open(image_file_path_or_image_data)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
    else:
        # Encode the image
        image = np.expand_dims(image_file_path_or_image_data, axis=0)
    # Initialize the caption with the start token
    caption = np.zeros((1, max_len))
    caption[0, 0] = word_index['<start>']

    # Generate the caption word by word
    for i in range(1, max_len):
        # Predict the next word
        predictions = model.predict([image, caption])
        index = np.argmax(predictions[0, i-1, :])
        caption[0, i] = index
        # Stop generating the caption if we've reached the end token
        if index == word_index['<end>']:
            caption[0, i] = index
            break
    # Convert the caption from word indices to words
    caption = [reverse_word_index[int(i)] for i in caption[0] if i > 0]
    # Return the caption as a string
    return ' '.join(caption)

def show_model_res(model, rows, cols, fig_size, word_index, reverse_word_index, val_images, val_set):
    # Define the figure and axis objects
    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    axs = axs.ravel()
    # Loop over the images and generate captions
    for i in range(rows * cols):
        print("predicting image index [{}/{}]:".format(int(i+1), rows * cols))
        # Generate a random index
        index = np.random.randint(len(val_images))
        # Get the image and caption
        image = val_images[index]
        # print(image)
        caption = generate_caption(model, image, word_index, reverse_word_index, True)

        for data in val_set:
            if np.array_equal(data['image'], image):
                true_caption = data['caption']
        # Plot the image and caption
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title('Predicted: {}\nTrue: {}'.format(caption, ' '.join(true_caption)))
        # Set the text color to green if the caption is correct, otherwise red
        if caption == ' '.join(true_caption):
            color = 'green'
        else:
            color = 'red'
        axs[i].title.set_color(color)
    # Adjust the spacing between the subplots
    fig.tight_layout()
    # Show the plot
    plt.show()

def show_train_res(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()




# 加载注释文件
with open(train_caption_file, 'r') as f:
    train_captions = json.load(f)['annotations']
with open(val_caption_file, 'r') as f:
    val_captions = json.load(f)['annotations']
# print("train_captions: "+"".join(str(i) for i in train_captions))
# print("val_captions: "+"".join(str(i) for i in val_captions))


# 创建字典将图片ID与其文件名对应
train_id_to_file = {}
for filename in os.listdir(train_image_dir):
    train_id_to_file[int(filename.split('.')[0])] = os.path.join(train_image_dir, filename)

val_id_to_file = {}
for filename in os.listdir(val_image_dir):
    val_id_to_file[int(filename.split('.')[0])] = os.path.join(val_image_dir, filename)

# 循环迭代字典中的所有值，并替换反斜杠为正斜杠
for key, value in train_id_to_file.items():
    train_id_to_file[key] = value.replace('\\', '/')
for key, value in val_id_to_file.items():
    val_id_to_file[key] = value.replace('\\', '/')



# 构建训练集和验证集
train_set = []
val_set = []
print("length of train_captions: " + str(len(train_captions)))
print("length of val_captions: " + str(len(val_captions)))

# 测试代码
test_train_len = int(train_percent * len(train_captions))
test_val_len = int(val_percent * len(val_captions))
BLOCK_NUM = 1000
train_captions_sub = random.sample(train_captions, test_train_len)
val_captions_sub = random.sample(val_captions, test_val_len)

# 预处理训练集
index = 0
for caption in tqdm(train_captions_sub, file=sys.stdout):
    # print(index)
    # 获取图片ID和文件名
    image_id = caption['image_id']
    file_path = train_id_to_file[image_id]
    # 加载图片并将其调整为相同的大小
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    # 将字节字符串转换为Unicode字符串
    caption_text = caption['caption'].encode('utf-8').decode('unicode_escape')
    # 将训练样本添加到列表中
    element = {'image': img, 'caption': caption_text}
    train_set.append(element)
    # print(element)
    index = index + 1

    # if index%BLOCK_NUM == BLOCK_NUM-1:
    #     print("train_data_pre_process [{}/{}]:".format(int(len(train_captions)/BLOCK_NUM), int(index/BLOCK_NUM)+1))

# 预处理验证集
index = 0
for caption in tqdm(val_captions_sub, file=sys.stdout):
    # print(index)
    # 获取图片ID和文件名
    image_id = caption['image_id']
    file_path = val_id_to_file[image_id]
    # 加载图片并将其调整为相同的大小
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    # 将字节字符串转换为Unicode字符串
    caption_text = caption['caption'].encode('utf-8').decode('unicode_escape')
    # 将验证样本添加到列表中
    element = {'image': img, 'caption': caption_text}
    val_set.append(element)
    index = index + 1
    # if index%1000 == 999:
    #     print("val_data_pre_process [{}/{}]:".format(int(len(train_captions)/BLOCK_NUM), int(index/BLOCK_NUM)+1))




# 创建词汇表
tokenizer = Tokenizer()
train_captions_text = [x['caption'] for x in train_set]
tokenizer.fit_on_texts(train_captions_text)

# 将单词序列转换为数字序列
train_sequences = tokenizer.texts_to_sequences(train_captions_text)
val_sequences = tokenizer.texts_to_sequences([x['caption'] for x in val_set])

# 准备图像数据
train_images = np.array([x['image'] for x in train_set])
val_images = np.array([x['image'] for x in val_set])

# 准备标签数据
max_length = max(len(x) for x in train_sequences)

# 将标签序列填充到相同的长度
train_captions = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_captions = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')

# 创建目标标签，将标签序列向右移动一个时间步
train_target = np.zeros((len(train_captions), max_length))
val_target = np.zeros((len(val_captions), max_length))
for i, caption in enumerate(train_captions):
    train_target[i, :-1] = caption[1:]
for i, caption in enumerate(val_captions):
    val_target[i, :-1] = caption[1:]

if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
if '<end>' not in tokenizer.word_index:
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1
# Define the word-to-index and index-to-word mappings
word_index = tokenizer.word_index
reverse_word_index = {v: k for k, v in word_index.items()}

# 将目标标签进行 one-hot 编码
num_words = len(tokenizer.word_index) + 1
# print("num_words:"+str(num_words))
# print(type(tokenizer.word_index))
print(tokenizer.word_index)
train_target = to_categorical(train_target.tolist(), num_words)
val_target = to_categorical(val_target.tolist(), num_words)
# print(train_target)
# print(val_target)



# Load pre-trained InceptionV3 model as the encoder
encoder_inputs = Input(shape=(224, 224, 3))
encoder = InceptionV3(weights='imagenet', include_top=False)
encoder_outputs = encoder(encoder_inputs)
encoder_outputs = Flatten()(encoder_outputs)
encoder_outputs = Dense(256, activation='relu')(encoder_outputs)

# Define the decoder using transformer architecture
decoder_inputs = Input(shape=(None,))
embedding = Embedding(num_words, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, dropout=0.5)(embedding)
decoder_lstm2 = LSTM(256, return_sequences=True, dropout=0.5)(decoder_lstm)
decoder_lstm3 = LSTM(256, return_sequences=True, dropout=0.5)(decoder_lstm2)
decoder_outputs = TimeDistributed(Dense(256, activation='relu'))(decoder_lstm3)

# Repeat the encoder output to match the decoder output shape
encoder_repeat = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, tf.shape(decoder_outputs)[1], 1]))(
    encoder_outputs)

# Combine the encoder and decoder
merged = Concatenate()([encoder_repeat, decoder_outputs])
merged = Dense(256, activation='relu')(merged)
merged = Dense(num_words, activation='softmax')(merged)
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=merged)
print(encoder_inputs.shape, decoder_inputs.shape)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005))
model.summary()


# Train the model
history = model.fit([train_images, train_captions], train_target, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=([val_images, val_captions], val_target), callbacks=[
        ModelCheckpoint('image_captioning_model.h5', save_best_only=True, save_weights_only=True)])
show_train_res(history)


# Evaluate the model on the validation set
res = model.evaluate([val_images, val_captions], val_target, verbose=0)
loss = res
# print('res: '+str(res))
print('Validation loss: %.3f' % loss)
# print('Validation accuracy: %.3f' % acc)

show_model_res(model, rows, cols, fig_size, word_index, reverse_word_index, val_images, val_set)
filepath = 'D:/ml/test_img/gwen.png'
print(generate_caption(model, filepath, word_index, reverse_word_index, False))

# 计算指标
BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, CIDEr, METEOR = \
    caculate_evaluation_indicator(model, val_set, word_index, reverse_word_index)

print("BLEU_1: ")
print(BLEU_1)
print("BLEU_2: ")
print(BLEU_2)
print("BLEU_3: ")
print(BLEU_3)
print("BLEU_4: ")
print(BLEU_4)
print("ROUGE: ")
print(ROUGE)
print("CIDEr: ")
print(CIDEr)
print("METEOR: ")
print(METEOR)
# print(SPICE)
print("data collection: ")
print(BLEU_1)
print(BLEU_2)
print(BLEU_3)
print(BLEU_4)
print(ROUGE)
print(CIDEr)
print(METEOR)






