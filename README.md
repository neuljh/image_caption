# image_caption
A simple Transformer model to finish image caption task.
# 这是一个非常简单的image_caption任务，如果有帮助希望你可以三连噢！！！
1.数据处理未使用GPU加速，因此速度很慢（能运行就万岁）
2.需要提前下载coco数据集到对应文件夹中，然后在主程序代码中修改coco数据集的路径：
![image](https://github.com/neuljh/image_caption/assets/132900799/dee3c41b-bcf4-4adb-b342-998e9ce47fba)
3.自定义你的参数：
![image](https://github.com/neuljh/image_caption/assets/132900799/0fcea603-643e-456b-ac34-3ca6f52415bd)
4.代码中有合理的注释可以参考。
5.详细步骤：
(1)初步完成
1)导入对应的库
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

2)申明必要的变量
# Define the maximum length of the caption
max_len = 30
# Define the number of rows and columns in the grid
rows = 5
cols = 1
# Define the figure size
fig_size = (10, 10)
# define some num
EPOCHS = 20
BATCH_SIZE = 32
# data size
train_percent = 0.03
val_percent = 0.1
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

3)重要函数说明
①　计算Image Caption模型的评估指标
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
用于计算机器学习模型的评估指标，接受4个参数：model（机器学习模型），val_set（验证集数据），word_index（词汇表）和reverse_word_index（反向词汇表）。
该函数首先从验证集中随机选取一定比例的数据，生成对应的参考和预测文本列表。然后将这些文本列表转换为参考和预测语料库，并计算BLEU、ROUGE、CIDEr、METEOR指标。
返回的是BLEU-1到BLEU-4、ROUGE、CIDEr和METEOR指标的值。
具体而言，这个函数首先从val_set中选取一定比例的数据，将其参考文本和预测文本分别放到两个列表中。接下来，将这些文本转换为参考和预测语料库。这些语料库将被用于计算各种评估指标。该函数使用nltk库中的Bleu、Rouge、Cider、Meteor和Spice类计算评估指标。在计算Bleu指标时，使用n-gram模型，其中n的值为4，即计算BLEU-1到BLEU-4分数。最后，该函数返回BLEU-1到BLEU-4、ROUGE、CIDEr和METEOR指标的值。
BLEU（Bilingual Evaluation Understudy）：这是一种基于n-gram匹配的指标，用于评估生成的文本与参考文本之间的相似度。BLEU越高，生成的文本越接近参考文本。
ROUGE（Recall-Oriented Understudy for Gisting Evaluation）：这是另一种基于n-gram匹配的指标，它在评估文本摘要方面非常流行。ROUGE评估生成的文本与参考文本之间的匹配程度。
CIDEr（Consensus-Based Image Description Evaluation）：这是一种基于多样性和一致性的指标，它考虑了多个参考文本，以及与众不同的词汇和短语的重要性。CIDEr越高，生成的文本越好。
METEOR（Metric for Evaluation of Translation with Explicit ORdering）：这是一种综合指标，它不仅考虑n-gram匹配，还考虑了生成的文本与参考文本之间的语法、语义和单词重叠度。METEOR可以更好地评估生成文本的整体质量。
SPICE（Semantic Propositional Image Caption Evaluation）：这是一种基于语义分析的指标，它评估生成的文本与参考文本之间的语义相似性。SPICE可以更好地评估生成文本的语义一致性。
WMD（Word Mover's Distance）：这是一种衡量词汇相似度的指标，它计算生成的文本与参考文本之间的词汇距离。WMD越低，生成的文本越好。

②　Generate captions for a sample image
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

该函数接受以下参数：
model: 训练好的神经网络模型，用于生成图像描述
image_file_path_or_image_data: 图像文件路径或图像数据（np.array），需要生成描述的图像
word_index: 字典，将单词转换为整数标识符
reverse_word_index: 字典，将整数标识符转换为单词
is_data: 如果是数据，则为True，否则为False
max_len: 生成的描述的最大长度
该函数的主要作用是生成输入图像的描述。函数首先根据输入的参数来加载图像数据，然后初始化一个描述，其中第一个单词是"<start>"。之后，函数将循环处理并预测每个单词，直到生成描述的最大长度或直到遇到结束标记"<end>"。最后，函数将生成的描述从单词的整数标识符转换为单词本身，并将其作为字符串返回。

③　展示模型验证集结果
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

用于显示图像及其对应的预测和真实文本标题。以下是该函数中各个参数和操作的详细解释：
model：神经网络模型。
rows：图像显示的行数。
cols：图像显示的列数。
fig_size：matplotlib 图像的尺寸。
word_index：字典，将单词映射到数字编码。
reverse_word_index：字典，将数字编码映射回单词。
val_images：用于验证的图像数据。
val_set：包含验证图像及其对应文本标题的数据集。
创建一个图像及其标题的网格，行数为 rows，列数为 cols，图像大小为 fig_size。对于每一个子图像，随机选择一个验证图像并生成对应的标题。从 val_set 中查找对应的真实标题。显示图像及其预测和真实标题，如果预测标题和真实标题相同，则标题显示为绿色，否则为红色。最后，调整子图像之间的间距并显示整个图像。

④　展示训练结果
def show_train_res(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
在函数中，训练集损失值（history.history['loss']）和验证集损失值（history.history['val_loss']）是从history参数中提取的，然后使用plt.plot()函数将它们绘制在同一个图表中。plt.title()用于设置图表的标题，plt.ylabel()和plt.xlabel()分别用于设置y轴和x轴的标签。plt.legend()函数用于设置图例，'Train'和'Test'表示训练集和验证集，loc='upper left'表示图例位于左上角。最后，plt.show()函数将图表显示出来。

4)主函数
# 加载注释文件
with open(train_caption_file, 'r') as f:
    train_captions = json.load(f)['annotations']
with open(val_caption_file, 'r') as f:
    val_captions = json.load(f)['annotations']
使用了 Python 中的 with 语句来打开两个 JSON 格式的文件并加载它们。with 语句用于在代码块结束后自动关闭文件。其中，变量 train_caption_file 和 val_caption_file 分别表示要打开的训练数据和验证数据的 JSON 文件名。
第一行代码中，open(train_caption_file, 'r') 打开名为 train_caption_file 的文件并指定以只读方式打开。然后，json.load(f) 将文件的内容加载到一个 Python 字典中，其中 f 是对打开文件的引用。接下来，使用字典键 annotations 获取文件中的 annotations 字段的值，并将其存储在 train_captions 变量中。
第二行代码中，open(val_caption_file, 'r') 打开名为 val_caption_file 的文件，并使用 json.load(f) 加载文件内容到一个 Python 字典中。同样，使用 annotations 键获取字典中的 annotations 字段的值，并将其存储在 val_captions 变量中。
总之，这段代码的作用是打开两个 JSON 文件，读取它们的内容，并将每个文件中的 annotations 字段的值分别存储在 train_captions 和 val_captions 变量中。这通常是用于加载训练和验证数据的一种常见方式，这些数据通常被存储在 JSON 或其他常用格式中。

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
首先创建两个空字典 train_id_to_file 和 val_id_to_file，用于存储训练集和验证集中的图像文件 ID 与其文件名的对应关系。其中，os.listdir(train_image_dir) 和 os.listdir(val_image_dir) 分别返回训练集和验证集图像文件所在目录中的所有文件名。
接着，通过循环遍历目录中的文件名，并用 int(filename.split('.')[0]) 的方式将文件名中的数字部分（即文件 ID）提取出来，将其作为键值存入字典 train_id_to_file 或 val_id_to_file 中，并将其对应的文件名作为值。
最后，通过循环迭代字典中的所有值，将字符串中的反斜杠 \ 替换为正斜杠 /，这样就能够避免在后续的代码中出现路径错误的问题。
最终，其目的是为了将训练集和验证集中的图像文件与其对应的文件名进行匹配。

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
这段代码的作用是构建训练集和验证集，并对它们进行预处理，以便将它们用于图像字幕生成模型的训练和验证。
首先，创建了两个空列表 train_set 和 val_set 用于存储预处理后的数据。然后通过打印信息，输出了训练集和验证集的长度。
接下来，对训练集和验证集进行预处理。首先通过随机抽样，将训练集和验证集分别缩小到一定的比例，以便更快地进行训练和验证。这里通过 random.sample 函数随机抽样，抽取了训练集中 test_train_len 个样本和验证集中 test_val_len 个样本。
然后，遍历训练集和验证集中的样本，并逐个进行预处理。每个样本包含图像和相应的字幕，因此需要对它们进行处理。对于每个样本，首先获取图像的 ID 和文件路径，并加载图像。然后将图像调整为相同的大小，这里的大小是 (224, 224)。接下来，将字节字符串转换为 Unicode 字符串，以便后续处理。最后将预处理后的样本以字典的形式添加到 train_set 或 val_set 中。

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

这段代码主要是为了准备图像标注数据集，其中涉及到的函数包括：
Tokenizer(): 一个类，用于将文本转化为单词序列，同时也可以进行词汇表的创建。
fit_on_texts(texts): 对文本进行处理，并生成词汇表。该方法可以接受一个字符串列表作为输入，其中每个字符串都是一个文本。
texts_to_sequences(texts): 将文本列表转换为数字列表，其中每个数字代表词汇表中的一个单词。
pad_sequences(sequences, maxlen, padding, truncating): 将数字序列填充到相同的长度。在这里，我们将序列进行填充和截断，使得它们的长度都等于最长序列的长度。
np.zeros(shape): 用于创建一个全是 0 的数组。
to_categorical(y, num_classes=None): 将整数转换为 one-hot 编码。
具体来说，这段代码的功能如下：
创建 Tokenizer 类的实例 tokenizer，并调用其 fit_on_texts 方法生成词汇表，这个词汇表中包含训练集中出现的所有单词。使用 tokenizer 对训练集和验证集中的标注文本进行数字化处理，得到 train_sequences 和 val_sequences，它们分别是训练集和验证集中所有标注文本的数字化表示。准备图像数据：从数据集中获取图像数据，并存储在 train_images 和 val_images 中。准备标签数据：首先计算出训练集中最长标注的长度 max_length，然后使用 pad_sequences 对 train_sequences 和 val_sequences 进行填充和截断，将它们的长度都变为 max_length，得到 train_captions 和 val_captions。接着，将 train_captions 和 val_captions 中的每个序列向右移动一个时间步，得到 train_target 和 val_target。对词汇表进行处理，将 '<start>' 和 '<end>' 加入词汇表中，并创建 word_index 和 reverse_word_index 两个字典，分别用于将单词转换为索引和将索引转换为单词。将 train_target 和 val_target 进行 one-hot 编码，得到 train_target 和 val_target，它们的形状都是 (样本数, 最长标注长度, 词汇表大小)。

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

这段代码定义了一个图像到文本的转换模型，使用了预训练的InceptionV3模型作为编码器(encoder)，以及一个Transformer解码器(decoder)。
下面是每个函数的解释：
Input函数：创建一个Keras张量，代表输入到模型的数据。这里创建了两个输入张量，一个用于图像(形状为(224,224,3))，一个用于文本序列(形状为(None,))。
InceptionV3函数：载入预训练的InceptionV3模型，该模型使用图像进行训练，并将图像编码成较小的特征向量。include_top=False参数指定不使用该模型的顶层分类器，因为我们要使用自己的解码器。
Flatten函数：将编码器输出的特征张量压扁成一个向量，方便后续处理。
Dense函数：创建一个全连接层，将编码器输出的特征向量映射到256维的向量。
Embedding函数：将文本序列中的单词映射到256维的向量，以便后续处理。
LSTM函数：创建一个LSTM层，以处理序列数据并生成隐藏状态。return_sequences=True参数指定LSTM层返回所有的输出，而不是只返回最后一个输出。dropout=0.5参数指定LSTM层的丢弃率为0.5，以防止过拟合。
TimeDistributed函数：将全连接层应用于序列中的每个时间步。这里，全连接层将生成一个256维的向量，以便后续处理。
Lambda函数：创建一个自定义操作，将编码器输出的特征向量重复多次，以匹配解码器的输出形状。
Concatenate函数：将编码器和解码器的输出连接在一起，形成一个更高维度的张量。
Dense函数：创建一个全连接层，将连接后的张量映射到256维的向量。
Dense函数：创建一个全连接层，将256维向量映射到单词数量的概率分布，以便选择最可能的单词作为输出。这里使用了softmax激活函数。
Model函数：创建一个Keras模型，将输入张量映射到输出张量。

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
这段代码是在编译模型。模型是神经网络中的一个重要部分，它定义了输入和输出之间的关系。编译模型是为了指定模型的训练方式和优化器。
loss='categorical_crossentropy' 表示使用分类交叉熵作为损失函数，用于衡量模型输出与真实标签之间的差异，即模型预测的结果与实际结果之间的误差。
optimizer=Adam(learning_rate=0.001) 表示使用Adam优化器，它是一种常用的梯度下降算法，用于更新模型中的权重参数。其中，learning_rate=0.001 表示学习率为0.001，即每一次更新权重参数时所采用的步长大小。

# Train the model
history = model.fit([train_images, train_captions], train_target, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=([val_images, val_captions], val_target), callbacks=[
        ModelCheckpoint('image_captioning_model.h5', save_best_only=True, save_weights_only=True)])
show_train_res(history)
这段代码的主要作用是训练一个图像描述生成模型，并将训练历史记录在 history 变量中。以下是每个参数的解释：
train_images: 训练图像的数据集
train_captions: 训练图像的文字描述的数据集
train_target: 训练目标的数据集，即模型预测的下一个单词
epochs: 模型训练轮数
batch_size: 模型每次训练时使用的数据批次大小
validation_data: 模型训练时的验证集数据，包含验证图像、文字描述和目标数据
callbacks: 模型训练时的回调函数，这里使用了一个 ModelCheckpoint 回调，每次在验证集上表现最好时保存模型权重到 image_captioning_model.h5 文件中，仅保存最佳模型权重，这个文件可以在后续的推理中加载模型。
show_train_res(history): 这个函数用于可视化训练过程中的历史记录，包括训练和验证集的损失和准确率等指标。history 变量包含了训练历史数据，包括训练损失、训练准确率、验证损失和验证准确率等。show_train_res 函数将这些指标绘制成图表，并在训练过程中实时更新。

# Evaluate the model on the validation set
res = model.evaluate([val_images, val_captions], val_target, verbose=0)
loss = res
print('Validation loss: %.3f' % loss)
show_model_res(model, rows, cols, fig_size, word_index, reverse_word_index, val_images, val_set)
filepath = 'D:/ml/test_img/gwen.png'
print(generate_caption(model, filepath, word_index, reverse_word_index, False))
# 计算指标
BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE, CIDEr, METEOR = \
    caculate_evaluation_indicator(model, val_set, word_index, reverse_word_index)
这段代码是一个基于深度学习的图像标注模型在验证集上的评估和指标计算。下面是函数的详细解释：
model.evaluate([val_images, val_captions], val_target, verbose=0)：使用验证集（val_images, val_captions）作为输入，计算模型在验证集上的损失值和其他指标，并返回一个结果对象 res。其中 val_target 是对应的标签。
loss = res：将损失值赋给变量 loss。
print('Validation loss: %.3f' % loss)：输出验证集上的损失值 loss。
show_model_res(model, rows, cols, fig_size, word_index, reverse_word_index, val_images, val_set)：显示模型在验证集上的结果。其中，show_model_res 是一个自定义函数，用于显示模型在验证集上的预测结果。
filepath = 'D:/ml/test_img/gwen.png'：设置一个测试图片路径。
generate_caption(model, filepath, word_index, reverse_word_index, False)：使用模型对测试图片进行标注，返回一个标注字符串。其中，generate_caption 是一个自定义函数，用于使用模型对图片进行标注。
caculate_evaluation_indicator(model, val_set, word_index, reverse_word_index)：计算模型在验证集上的评估指标。其中，caculate_evaluation_indicator 是一个自定义函数，用于计算模型的评估指标，包括 BLEU-1~4、ROUGE、CIDEr 和 METEOR 等指标。函数返回一个元组，包含所有指标的值。

5)实验结果
①　Epoch=10，batch_size=32,train_percent = 0.02,val_percent = 0.1,
②　score_percent = 0.01
计算输出了训练集和验证集的大小：
![image](https://github.com/neuljh/image_caption/assets/132900799/63445033-119a-4609-81a6-5ee6c21b6bd0)
经过测试，发现本电脑在加载超过4w条训练集会发生内存爆炸的情况，因此这里仅仅使用部分数据集，这里加载11835条训练集和2501条验证集：
![image](https://github.com/neuljh/image_caption/assets/132900799/7a32430d-50f7-474b-9afb-8fbae3940230)

得到的词汇表输出如下：
![image](https://github.com/neuljh/image_caption/assets/132900799/f71ee399-8b7e-4d25-b0cd-48812bea191c)
![image](https://github.com/neuljh/image_caption/assets/132900799/0eea2820-41aa-44ac-8c0b-11f6c57fbb52)

模型结构：
![image](https://github.com/neuljh/image_caption/assets/132900799/ecfe16a7-24f7-4fa6-86a7-efa7b8c933bd)
![image](https://github.com/neuljh/image_caption/assets/132900799/ad286fda-7afb-443a-b71d-02d05255d932)


模型训练过程：
![image](https://github.com/neuljh/image_caption/assets/132900799/f6f76deb-b4b5-4dc9-b292-cf0d6e7e2de2)

验证集损失：
![image](https://github.com/neuljh/image_caption/assets/132900799/0fb298c6-cfb9-4438-b925-81b292634a28)

自定义加载的图片进行image caption，原图片和模型结果如下：
![image](https://github.com/neuljh/image_caption/assets/132900799/95c9b1ad-6279-4886-858a-9a23226c0daa)
![image](https://github.com/neuljh/image_caption/assets/132900799/a54ae38c-f47c-4902-98cc-8c0cfb42314f)


随机抽取25个验证集数据进行模型评估：
![image](https://github.com/neuljh/image_caption/assets/132900799/e4234696-06cc-4727-992d-9553b3c3e9e5)

最终得到的评估结果：
![image](https://github.com/neuljh/image_caption/assets/132900799/d7992ffe-2dd0-41d9-952e-bbca7ed248c5)

输出得到的loss变化趋势图：
![image](https://github.com/neuljh/image_caption/assets/132900799/fbc84cab-d6c8-4cda-b955-1496f8e4f449)

图片的真实标题和预测标题：
![image](https://github.com/neuljh/image_caption/assets/132900799/d332c784-db5f-4fcd-b871-d5601f1db416)

可以看出训练效果还是比较一般的。
③　Epoch=20，batch_size=32,train_percent = 0.03,val_percent = 0.1,
④　score_percent = 0.01
评估结果：
![image](https://github.com/neuljh/image_caption/assets/132900799/f52b2dae-1251-4bf5-90c7-877405e63792)

输出得到的loss变化趋势图：
![image](https://github.com/neuljh/image_caption/assets/132900799/f3d4b266-6590-4a0f-ba38-4227d77a40df)

图片的真实标题和预测标题：
![image](https://github.com/neuljh/image_caption/assets/132900799/8ae28678-bc7d-49fc-b8e4-aa1ae8aed725)

验证集损失：
![image](https://github.com/neuljh/image_caption/assets/132900799/8b95c501-2410-4172-9780-412cb866cd99)
