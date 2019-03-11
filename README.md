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
https://wikidocs.net/22660 참고해서 하나의 txt 파일로 구성

## Training
```
python train_w2v.py --dataset "txt 파일위치"
```
### t-sne 그림
```
python t_sne.py --model "path of the model" --word "키워드" --list ["임의","선택", "단어","들"]
```
![역사 t-sne](images_readme/역사_t_sne.png)
![스마트폰 t-sne](images_readme/역사_t_sne.png)
