from konlpy.tag import Okt
import codecs
import gensim
import multiprocessing
import time
import logging
import argparse

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    cores = multiprocessing.cpu_count()
    
    parser = argparse.ArgumentParser(description='Korean word2vec using okt')
   
    parser.add_argument("--dataset", dest = 'dataset', help = "text data for embedding")
    parser.add_argument("--min_count", dest = 'min_count', help = "minimum count", default = 5)
    parser.add_argument("--window", dest = 'window', help = "w2v window", default = 2)
    parser.add_argument("--size", dest = 'size', help = "w2v embedding size", default = 300)
    parser.add_argument("--sample", dest = 'sample', default = 6e-5)
    parser.add_argument("--alpha", dest = 'alpha', default = 0.03)
    parser.add_argument("--min_alpha", dest = 'min_alpha', default = 0.0007)
    parser.add_argument("--negative", dest = 'negative', default = 20)
    parser.add_argument("--workers", dest = 'workers', help = "number of workers", default = cores-1)
    parser.add_argument("--savename", dest = 'savename', default = "korw2v")
    parser.add_argument("--epoch", dest='epoch', default=30)
    return parser.parse_args()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

def report(text, index):
    if index % int(len(text)*0.001) == 0:
        print('{}/{} ({}%) completed'.format(index,len(text), index / len(text) * 100))

def tokenize(line, index, tokenizer, report=None):
    if report:
        report(text, index) 
    return tokenizer.morphs(line)


if __name__=='__main__':
    
    args = arg_parse()
    okt=Okt()
    
    print("tokenizing the text")
    t = time.time()
    #토큰화 과정, 148분 걸렸음, there may be a faster way
    with open(args.dataset, 'r') as f:
         text = f.readlines() #6010095줄
    token = [tokenize(line, index, okt, report) for index, line in enumerate(text)]

    print('Time to tokenize wikipedia text: {} mins'.format(round((time.time() - t) / 60, 2)))

    embedding = gensim.models.word2vec.Word2Vec(min_count=args.min_count, window=args.window, size=args.size, sample=args.sample,
                                                alpha=args.alpha, min_alpha=args.min_alpha, negative=args.negative, workers=args.workers)
    
    
   
    t = time.time()
    embedding.build_vocab(token, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))
    
    
    print("training the model")  
   
    ##training the model
    t = time.time()
    embedding.train(token, total_examples=embedding.corpus_count, epochs=args.epoch, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))
    
    #As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient:

    embedding.init_sims(replace=True)
    embedding_name = args.savename
    embedding.save(embedding_name)
    print("saved the models as {}".format(embedding_name))