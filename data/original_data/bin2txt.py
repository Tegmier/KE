from gensim.models import KeyedVectors

binpath = r"E:\code\KE\data\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin"
txtpath = r"E:\code\KE\data\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.txt"
# 加载Google's Word2Vec模型
model = KeyedVectors.load_word2vec_format(binpath, binary=True)

# 保存为文本文件
model.save_word2vec_format(txtpath, binary=False)
