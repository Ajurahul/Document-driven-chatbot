import os

def load_document(dir):
    docs = []
    for filename in os.listdir(dir):
        if(filename.endswith(".txt")):
            with open(os.path.join(dir, filename) , 'r')as file:
                      docs.append(file.read())
    return docs

def preprocess_text(text):
    return text.strip().replace("\n", " ")
def preprocess():
    docs = load_document("/docs")
    preprocessed = [preprocess_text(doc) for doc in docs]
    return preprocessed