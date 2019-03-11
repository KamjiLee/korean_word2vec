##t-sne visualization

import gensim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import pandas as pd
import argparse
from sklearn.manifold import TSNE

sns.set_style("darkgrid")


def arg_parse():
    """
    Parse arguements to the detect module
    
    """    
    parser = argparse.ArgumentParser(description='t-sne plot')
   
    parser.add_argument("--keyword", dest = 'keyword', type=str, default=None, help = "t-sne plot의 중심 단어")
    parser.add_argument("--embedding", dest = 'embedding', default=None, help = "저장된 w2v embedding 파일 경로")
    parser.add_argument("--list", dest = 'list', nargs='+', type=str, default=[], help = "임의로 설정해서 plot할 단어들의 list")
    return parser.parse_args()


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(arrays)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.savefig('{}_t_sne.png'.format(word))
    print(model.wv.most_similar(word))
    
if __name__=='__main__':
    
    args=arg_parse()
    mpl.rc('font',family='NanumGothic') #우분투에서 한글 font 문제 해결, windows는 다름
    model = gensim.models.Word2Vec.load(args.embedding)
    word = args.keyword
    list_names = args.list
    
    tsnescatterplot(model,word,list_names)