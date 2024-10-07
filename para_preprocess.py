import os
import nltk
import pickle
import random
from transformers import BertTokenizer
from nltk import word_tokenize
import editdistance

def loadpkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    return obj

class ParaNMTPreprocessor:
    def __init__(self, data_path="path/para"):
        self.data_path = data_path
        self.word_to_index = {}  # Store word-to-index mapping
        self.index_to_word = {}  # Store index-to-word mapping

    def create_vocabulary(self, min_count=3, include_validation=True):
        word_counts = {} 
        train_src_file = os.path.join(self.data_path, 'train_src.txt')
        train_trg_file = os.path.join(self.data_path, 'train_trg.txt')
        if include_validation:
            valid_src_file = os.path.join(self.data_path, 'valid_src.txt')
            valid_trg_file = os.path.join(self.data_path, 'valid_trg.txt')

        # Create vocabulary from train and optionally valid data
        for file in [train_src_file, train_trg_file]:
            with open(file,encoding='utf-8') as f:
                for line in f.readlines():
                    words = nltk.word_tokenize(line)
                    words = [word.lower() for word in words]
                    tags = list(zip(*nltk.pos_tag(words)))[1]
                    for word in words:
                        if word not in word_counts:
                            word_counts[word] = 0
                        word_counts[word] += 1
                    for tag in tags:
                        if tag not in word_counts:
                            word_counts[tag] = 0
                        word_counts[tag] += 1

        if include_validation:
            for file in [valid_src_file, valid_trg_file]:
                with open(file, encoding='utf-8') as f:
                    for line in f.readlines():
                        words = nltk.word_tokenize(line)
                        words = [word.lower() for word in words]
                        tags = list(zip(*nltk.pos_tag(words)))[1]
                        for word in words:
                            if word not in word_counts:
                                word_counts[word] = 0
                            word_counts[word] += 1
                        for tag in tags:
                            if tag not in word_counts:
                                word_counts[tag] = 0
                            word_counts[tag] += 1

        self.word_to_index = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
        index = 4
        # Create word to index mapping
        for word, count in word_counts.items():
            if count >= min_count:
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    index += 1

        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        # Save the vocabulary
        with open("data2/word2idx2.pkl", 'wb') as f:
            pickle.dump(self.word_to_index, f)
        with open("data2/idx2word2.pkl", 'wb') as f:
            pickle.dump(self.index_to_word, f)

    def convert_to_indices(self, sub_file="/train"):
        word_to_index = self.word_to_index
        src_file = os.path.join("para" + sub_file + "_src.txt")
        trg_file = os.path.join("para" + sub_file + "_trg.txt")
        src_pos_indices = []
        trg_pos_indices = []
        src_word_indices = []
        trg_word_indices = []

        with open(src_file, encoding='utf-8') as f:
            for line in f.readlines():
                words = nltk.word_tokenize(line)
                words = [word.lower() for word in words]
                word_indices = [word_to_index[word] if word in word_to_index else word_to_index['UNK'] for word in words]
                tags = list(zip(*nltk.pos_tag(words)))[1]
                tag_indices = [word_to_index[tag] if tag in word_to_index else word_to_index['UNK'] for tag in tags]
                src_pos_indices.append(tag_indices)
                src_word_indices.append(word_indices)

        with open(trg_file, encoding='utf-8') as f:
            for line in f.readlines():
                words = nltk.word_tokenize(line)
                words = [word.lower() for word in words]
                word_indices = [word_to_index[word] if word in word_to_index else word_to_index['UNK'] for word in words]
                tags = list(zip(*nltk.pos_tag(words)))[1]
                tag_indices = [word_to_index[tag] if tag in word_to_index else word_to_index['UNK'] for tag in tags]
                trg_pos_indices.append(tag_indices)
                trg_word_indices.append(word_indices)

        with open("data2" + sub_file + "/src_pos.pkl", 'wb') as f:
            pickle.dump(src_pos_indices, f)
        with open("data2" + sub_file + "/trg_pos.pkl", 'wb') as f:
            pickle.dump(trg_pos_indices, f)
        with open("data2" + sub_file + "/src.pkl", 'wb') as f:
            pickle.dump(src_word_indices, f)
        with open("data2" + sub_file + "/trg.pkl", 'wb') as f:
            pickle.dump(trg_word_indices, f)

        print(len(src_word_indices))
        print(len(trg_word_indices))
        print(src_word_indices[0])
        print(trg_word_indices[0])

    def generate_bert_ids(self, sub_file="/test"):
        index_to_word = self.index_to_word
        input_file = "data2" + sub_file + "/src.pkl"
        output_file = "data2" + sub_file + "/trg.pkl"

        input_sequences = loadpkl(input_file)
        output_sequences = loadpkl(output_file)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_src_texts = [' '.join([index_to_word[word] for word in sent]) for sent in input_sequences]
        bert_trg_texts = [' '.join([index_to_word[word] for word in sent]) for sent in output_sequences]
        bert_src_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in bert_src_texts]
        bert_trg_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in bert_trg_texts]

        with open("data2" + sub_file + "/bert_src.pkl", 'wb') as f:
            pickle.dump(bert_src_ids, f)
        with open("data2" + sub_file + "/bert_trg.pkl", 'wb') as f:
            pickle.dump(bert_trg_ids, f)

    def find_exemplars(self, sub_file="/test"):
        pos_tags = loadpkl("data2" + sub_file + "/trg_pos.pkl")
        target_sentences = loadpkl("data2" + sub_file + "/trg.pkl")

        similar_sentences = []
        for i in range(len(target_sentences)):
            sentence_length = len(target_sentences[i])
            similarity_scores = [100 for _ in range(len(target_sentences))] 

            if i % 1000 == 0:
                print(i)

            for j in range(max(0, i - 20000), min(i + 20000, len(target_sentences))):
                if i != j:
                    other_sentence_length = len(target_sentences[j])
                    if abs(sentence_length - other_sentence_length) > 1:
                        continue
                    if len(list(set(target_sentences[i]) & set(target_sentences[j]))) + 2 > len(list(set(target_sentences[i]))):
                        continue
                    current_pos_tags = pos_tags[i]
                    other_pos_tags = pos_tags[j]
                    edit_distance = editdistance.eval(current_pos_tags, other_pos_tags)
                    similarity_scores[j] = edit_distance

            min_distance = min(similarity_scores)
            similar_indices = [i for i, score in enumerate(similarity_scores) if score == min_distance]

            similar_sentences.append(random.sample(similar_indices, min(5, len(similar_indices))))

        with open("data2" + sub_file + "/sim.pkl", 'wb') as f:
            pickle.dump(similar_sentences, f)
        print(similar_sentences)

# Example usage
processor = ParaNMTPreprocessor()
processor.create_vocabulary(include_validation=True)
processor.convert_to_indices(sub_file="/train")
processor.convert_to_indices(sub_file="/valid")
processor.convert_to_indices(sub_file="/test")
processor.find_exemplars(sub_file="/train")
processor.find_exemplars(sub_file="/valid")
processor.find_exemplars(sub_file="/test")
processor.generate_bert_ids(sub_file="/train")
processor.generate_bert_ids(sub_file="/valid")
processor.generate_bert_ids(sub_file="/test")

