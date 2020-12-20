import sys
from Corpus import Corpus

corpus = Corpus() # khởi tạo
dirr = "resources/corpus_mini.txt"  # chọn đường dẫn corpus
corpus.read(dirr)  # đọc đường dẫn
corpus.preprocess()   # Tiền xử lí
corpus.read_word2vec()  # Đọc model word2vec để tìm từ đồng nghĩa
corpus.read_ner()   # Đọc dữ liệu đã xử lí tên thực thể và từ loại

# Từ đồng nghĩa. TOp 10 kèm với cosine similarity. Ko tìm thấy thì return list rỗng []
print(corpus.find_similar('trung'))

# Tìm kiếm nhập nhằng (kí tự, từ, tiếng). case = 0 --> non-case sensitive, case = 1 --> case sensitive
temp = corpus.search_ambiguous('ng', case = 1)
for res in temp[:10]:
    print(res)

# Tìm theo tên thực thể. VD: tìm người tên việt (I- inside, B- begining). Ko thấy trả rỗng
temp = corpus.search_by_ner('Việt', 'I-PER')
for res in temp:
    print(res)

# Tìm theo loại từ. Ko thấy trả rỗng
temp = corpus.search_by_pos('Việt', 'N')
for res in temp:
    print(res)







