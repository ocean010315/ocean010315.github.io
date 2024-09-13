---
title: 한국어 형태소 분석기, tokenizer
date: 2024-09-13 00:00:00 +09:00
categories: [NLP, 전처리]
tags: [NLP, tokenizer]
use_math: true
---

한국어는 어미와 조사를 포함한 접사가 결합된 형태로 문법적 기능을 수행한다. 따라서 subword tokenization을 수행할 때 형태소 분석만을 수행하기 보다 추가로 BPE 알고리즘을 적용하는 등 한국어에 특화된 적절한 전처리 과정이 필요하다.  
또한 인터넷의 발전으로 크롤링을 통해 데이터를 수집하는 경우, 'ㅋㅋㅋ', '아앗..'과 같이 문법적으로 맞지 않는 언어가 의미를 갖기도 한다. 아래와 같이 자음으로만 이루어진 문자인데도 그 개수에 따라 의미하는 바가 다른 것처럼 사람과 비슷한 수준의 언어를 구사하는 모델을 위해서는 이러한 글자 하나 하나 소중히 고려해야 한다.  
<img src="/assets/img/korean_lol.jpg" width="500">  
<figcaption style="text-align:center; font-size:15px; color:#808080; margin-top:0px">출처: <a href="https://www.chosun.com/site/data/html_dir/2020/04/10/2020041002782.html">중앙일보</a></figcaption>  

## 한국어 형태소 분석기
한국어를 대상으로 형태소 분석을 수행하는 가장 쉽고 편한 방식은 KoNLPy이다. `Hannanum`, `Kkma`, `Komoran`, `Mecab`, `Okt`와 같은 여러 형태소 분석기가 포함되어 있다. 요즘은 Hugging Face에서 한국어 버전 모델을 제공하고 그에 맞는 tokenzer까지 세트로 제공하기 때문에 형태소 분석기 자체를 쓸 일이 없을 것이다. 하지만 비교적 넓은 범위의 형태소 분석을 수행하는 Okt와 일부 모델들의 tokenzier에 기반이 되는 Mecab-ko는 살펴보고 넘어가자.

### Mecab-ko
원래 일본어 기본의 tokenizer인데, 한국인 개발자가 한국어 형태소 분석기로 발전시켰다. 앞서 언급한 형태소 분석기들보다 속도가 빠르고 성능 또한 준수하여 가장 많이 사용되는 형태소 분석기이다.

### Okt
SNS(구 twitter)에서 수집한 데이터를 기반으로 형태소 분석을 수행한다. SNS 기반이라 'ㅋㅋ', '^^', 'ㅠㅠ'와 같은 글자들도 처리할 수 있다고 알고 있었는데, 다른 형태소 분석기의 경우 \<Unknown>으로 처리하는 것을 \<Korean Partical>으로, 쉼표나 마침표 등을 \<Punctuation>으로 처리하는 등의 차이만 있는 듯하다.  

두 형태소 분석기 중 문맥을 이해하고 그에 맞는 품사 태깅을 하는 것은 Mecab-ko이다. [공식 문서](https://konlpy.org/ko/latest/morph/#comparison-between-pos-tagging-classes)를 살펴보면 **아버지가방에들어가신다**라는 문장으로 품사 태깅을 수행한 결과, Mecab-ko는 **아버지(NNG) 가(JKS) 방(NNG) 에(JKB) 들어가(VV) 신다(EP+EC)**로 적절히 구분한 반면, Okt는 **아버지(Noun) 가방(Noun) 에(Josa) 들어가신(Verb) 다(Eomi)**라고 구분한다.  
또한 **나는 밥을 먹는다 vs. 하늘을 나는 자동차**의 **나**를 구분하는 task에서는 Mecab-ko의 경우 전자를 명사로, 후자를 동사로 잘 구분하는 반면, Okt의 경우 전자와 후자를 모두 명사로 구분한다.  
공식 문서가 너무 옛날 버전이라 직접 수행해본 결과, Okt가 **아버지가방에들어가신다**는 잘 수행하지만, 여전히 **나**의 차이는 구분하지 못한다. 업데이트가 되긴 했지만 아직 부족한 점이 보이는 모습이다.

## 한국어 Tokenizer

### KLUE/RoBERTa tokenizer
GLUE 데이터셋과 유사하게 STS, NLI 등 다양한 NLP task를 수행할 수 있는 한국어 벤치마크 데이터셋이다. 논문 [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680)의 **Tokenization** 부분을 보면 두 단계로 tokenization을 수행했다.  
1. 형태소 분석기 Mecab-ko를 사용하여 raw data를 형태소 단위로 구분  
2. Huggingface Tokenizer의 wordpiece tokenizer를 사용해서 BPE 알고리즘 적용  

이렇게 형태소 분석을 수행한 후 BPE와 같은 tokenization을 추가로 수행하면 앞서 'ㅋㅋ'와 같이 \<Unknown>으로 처리되는 token들까지도 의미를 갖는 한 단위로 간주될 수 있다.

### KoBERT tokenizer
SentencePiece tokenizer를 사용한다. 공백을 고려하지 않고 전체 텍스트를 하나의 문자열로 취급하기 때문에 공백이 명확하지 않은 교착어들에서도 잘 작동한다.  

순위가 높은 모델들의 tokenizer를 알아보고 비교해보는 과정을 수행해봐야겠다.

> **출처**  
[KoNLPy](https://konlpy.org/ko/latest/morph/#comparison-between-pos-tagging-classes)  
[KLUE](https://arxiv.org/pdf/2105.09680)
