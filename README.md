# Korean_word2vec

word2vec in korean

### Prerequisites

```
python
gensim
```
### Installing


한국어 위키 덤프 파일 다운로드

```
wget https://dumps.wikimedia.org/kowiki/20190301/kowiki-20190301-pages-articles.xml.bz2
```

위키피디아 덤프 파일 파싱

```
git clone "https://github.com/attardi/wikiextractor.git"
python WikiExtractor.py kowiki-20190301-pages-articles.xml.bz2 
```

End with an example of getting some data out of the system or using it for a little demo

## Training
```
python train_w2v.py
```
### t-sne 그림
```
python t_sne.py --model "path of the model" --word "키워드" --list ["임의","선택", "단어","들"]
```


 
