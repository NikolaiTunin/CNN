def load_dataset():
    with open('dataset.txt', 'r') as data_file:
        return data_file.readlines()


text_data = load_dataset()

from math import log10


def take_terms(data_array):
    terms = []
    line_terms = []
    term = ''
    for line in data_array:
        for char in line:
            if (ord(char) >= 65) & (ord(char) <= 122) | (ord(char) == 39):
                term += char.lower()
            else:
                if term != '':
                    line_terms.append(term)
                    term = ''
        terms.append(line_terms)
        line_terms = []
    return terms


def take_unique_terms(terms):
    unique_terms = []
    for line_terms in terms:
        for term in line_terms:
            if unique_terms.count(term) == 0:
                unique_terms.append(term)
    return unique_terms


def build_term_frequency(terms, unique_terms):
    terms_frequencies = []
    terms_in_doc_frequencies = []
    for doc in terms:
        L = len(doc)
        for u_term in unique_terms:
            if L != 0:
                tf = doc.count(u_term) / L
            else:
                tf = 0
            terms_in_doc_frequencies.append(tf)
        terms_frequencies.append(terms_in_doc_frequencies)
        terms_in_doc_frequencies = []
    return terms_frequencies


def build_inverse_term_frequency(terms, unique_terms):
    L = len(terms)
    idfs = []
    count = 0
    for u_term in unique_terms:
        for doc in terms:
            if doc.count(u_term) != 0:
                count += 1
        idf = log10(L/count)
        idfs.append(idf)
        count = 0
    return idfs


def build_tf_idf(terms_frequencies, idfs):
    tf_idf_docs = []
    tf_idfs = []
    for doc in terms_frequencies:
        for i in range(len(doc)):
            tf_idf = doc[i] * idfs[i]
            tf_idfs.append(tf_idf)
        tf_idf_docs.append(tf_idfs)
        tf_idfs = []
    return tf_idf_docs


def main():
    """
    print(len(text_data))

    #выбираем термы из коллекции документов
    terms = take_terms(text_data)

    #составляем список уникальных термов в коллекции
    unique_terms = take_unique_terms(terms)

    #строим векторное отображение коллекции документов по принципу TF (Term Frequency)
    terms_frequencies = build_term_frequency(terms, unique_terms)

    #проверяем размерность полученной коллекции
    print(len(terms_frequencies))
    print(len(terms_frequencies[0]))

    #проверяем точность
    index = unique_terms.index('in')
    print('Термы в 1 документе:', terms[0])
    print('Количество термов в 1 документе:', len(terms[0]))
    print('Частота вхождения терма in в 1 документ:', str(terms_frequencies[0][index]))

    index = unique_terms.index('the')
    print('\nТермы в 1238 документе:', terms[1237])
    print('Количество термов в 1238 документе:', len(terms[1237]))
    print('Частота вхождения терма the в 1238 документ:', str(terms_frequencies[1237][index]))

    #строим idf
    idfs = build_inverse_term_frequency(terms, unique_terms)

    #строим векторное отображение коллекции документов по принципу TF-IDF
    tf_idf_docs = build_tf_idf(terms_frequencies, idfs)

    #проверяем
    print('Термы в 1 документе:', terms[0])
    index = unique_terms.index('in')
    print('Метрика tf-idf для терма in в 1 документе', str(tf_idf_docs[0][index]))
    """



main()