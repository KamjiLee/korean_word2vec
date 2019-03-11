# Korean_word2vec

word2vec in korean

### Prerequisites

```
python3
gensim
konlpy
codecs
matplotlib
seaborn
sklearn
pandas
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
### 임베딩 결과
```
print(embedding.wv.most_similar(positive=["공포"]))
print(embedding.wv.most_similar(positive=["동물"]))
print(embedding.wv.most_similar(positive=["통계"]))

#[('제정', 0.4467240869998932), ('두려움', 0.4429143965244293), ('혐오', 0.4345221519470215), ('불안감', 0.4040243923664093), ('栱包', 0.402253657579422), ('증오', 0.38317736983299255), ('분노', 0.3810647130012512), ('스릴러', 0.3755665123462677), ('질린다', 0.3750323951244354), ('치안유지법', 0.3747129440307617)]
#[('생물', 0.7137835621833801), ('곤충', 0.6817206144332886), ('짐승', 0.6774595379829407), ('식물', 0.6387141942977905), ('포유류', 0.626631498336792), ('포유동물', 0.6182601451873779), ('물고기', 0.610832929611206), ('동식물', 0.609168529510498), ('척추동물', 0.6020264625549316), ('인간', 0.5923241376876831)]
#[('통계학', 0.5518889427185059), ('분석', 0.5027167797088623), ('자료', 0.4961372911930084), ('수치', 0.48129206895828247), ('센서스', 0.47838157415390015), ('정보', 0.4725317060947418), ('메타분석', 0.4574437737464905), ('조사', 0.4519674777984619), ('연구', 0.44756704568862915), ('기준', 0.44388657808303833)]
```


### t-sne 그림
```
python t_sne.py --model "path of the model" --word "키워드" --list ["임의","선택", "단어","들"]
```
![역사 t-sne](readme_images/역사_t_sne.png)
![스마트폰 t-sne](readme_images/스마트폰_t_sne.png)
