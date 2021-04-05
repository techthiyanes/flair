import xml.etree.ElementTree as ET
from flair.data import Sentence, Corpus
from flair.datasets import SentenceDataset

def extract_XML(path):
    data = []
    tree = ET.parse(path)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        text = sentence.find("text").text
        flair_sentence = Sentence(text)
        for token in flair_sentence:
            token.set_label("polarity", "O")
        aspectTerms = sentence.find("aspectTerms")
        if aspectTerms:
            for aspectTerm in aspectTerms:
                _from = int(aspectTerm.get('from'))
                _to = int(aspectTerm.get('to'))
                term = aspectTerm.get("term")
                polarity = aspectTerm.get("polarity")
                _curr_from = 0
                _curr_to = 0
                for token in flair_sentence:
                    _curr_to += len(token.text) + 1
                    if _curr_from - len(term) < _from < _curr_from + len(term):
                        if _curr_to - len(term) < _to < _curr_to + len(term):
                            if term.__contains__(token.text):
                                token.set_label("polarity", polarity)
                    _curr_from = _curr_to

        data.append(flair_sentence)

    return data

def main():
    laptop_data = extract_XML('aspect_data/Laptop_Train_v2.xml')
    restaurant_data = extract_XML('aspect_data/Restaurants_Train_v2.xml')

    laptop_corpus = Corpus(SentenceDataset(laptop_data))
    restaurant_corpus = Corpus(SentenceDataset(restaurant_data))

    print()

if __name__ == "__main__":
    main()