---
title: 한국어 STS 모델
date: 2024-09-18 00:00:00 +09:00
categories: [NLP, 모델]
tags: [NLP, STS, LLM]
use_math: true
---

### STS란?
STS란 Semantic Textual Similarity의 이니셜을 딴 이름으로, 두 문장이 주어졌을 때 의미가 얼마나 유사한지를 파악하는 task이다.  
label은 0-5점의 값을 가지며, 2.5점을 기준으로 0-2.4까지는 False, 2.5-5.0까지는 True값을 가진다.  
해당 task는 검색 엔진 사용 시 유사한 내용을 함께 검색해서 결과를 보여주는 기술이나, 중복된 문장이나 내용을 감지하는 표절 시스템, 번역 품질 평가 등 현재 생성형 AI가 활용되는 다양한 분야에 적용될 수 있다.  

### 한국어 STS 데이터셋
**KLUE**  
한국어 기반 LLM 사전 학습 데이터셋에는 대표적으로 KLUE가 있다. [Hugging Face](https://huggingface.co/datasets/klue/klue)에서도 찾아볼 수 있으니 확인해보기!  
[논문](https://arxiv.org/pdf/2105.09680)은 아주 긴데, STS 부분만 살펴보면, 소스는 `AIRBNB`, `POLICY`, `PARAKQC`로 총 세 개이다. 각각 에어비앤비 리뷰(구어체), 정치 뉴스, 스마트홈에 대한 질의응답(구어체)으로 구성된 데이터셋이다.  

**KorSTS**  
Kakao Brain에서 만든 한국어 STS 데이터셋이다.

### 한국어 STS 모델
KLUE를 구축하고 학습해서 비교 실험을 수행한 결과는 [KLUE github](https://github.com/KLUE-benchmark/KLUE?tab=readme-ov-file)에 있다. 확인해보고 적절한 모델을 골라도 좋을 듯하다.  

**RoBERTa**  
대표적으로 앞서 설명한 KLUE 데이터셋으로 사전 학습을 수행한 RoBERTa가 있다. BERT가 영어 기반이기 때문에 한국어와는 맞지 않아 보다 더 잘 학습될 수 있는 환경에서 한국어 데이터를 사용하여 학습한 모델이다. 
1. BERT보다 더 많은 데이터셋으로 더 오래 학습
2. NSP 학습 삭제(어디선가 NSP는 별로 의미 없고 MLM만이 효과가 있다고 들었던 것 같은데 출처가 뭐더라...)
3. 더 긴 문장들로 학습
4. mask를 epoch마다 중복되지 않도록 dynamic하게 변경(엄청난 장점이라고 하는데 언젠가 논문을 보게 되면 정리하는 걸로)  

**KLUE BERT base**  
RoBERTa를 학습할 때 BERT base까지도 함께 학습하여 비교 실험을 수행하였다. 결과는 RoBERTa-large가 가장 좋은 성능을 보였지만 사용하는 데이터셋이 무엇인지에 따라 해당 모델도 비교해보면 좋을 것 같다.  

**XLM-R**  
XLM은 cross-lingual language model의 약자로, 다국어를 목표로 사전 학습을 수행한 BERT를 XLM이라고 한다. 단일 언어 및 병렬 데이터셋을 사용하는데, 예를 들어, 영어로 된 문장이 있으면 같은 문장의 프랑스어 버전을 한 쌍으로 하는 데이터셋이 구성된다. 이를 교차 언어 데이터셋이라고 한다. XLM-R은 XLM-RoBERTa로 교차 언어 학습을 수행한 것으로, 한국어에 보다 더 잘 맞는 모델이다.

**KoELECTRA**  
generator에서 나온 token을 보고 discriminator에서 `real` token인지, `fake` token인지 판별하는 GAN의 방식으로 학습한다. 모든 input token에 대해서 분석할 수 있으며, BERT 등과 비교했을 때 보다 좋은 성능을 보인다고 한다. RoBERTa와 달리 tokenizer에 Mecab-ko 형태소 분석기를 적용하지 않았고, WordPiece 알고리즘을 사용했다.  

**KF-DeBERTa**  
카카오뱅크 & 에프엔가이드에서 DeBERTa-v2를 기반으로 학습한 한국어 버전 DeBERTa이다. 범용 도메인과 금융 도메인에 대해서 학습하였고, 리드미에서 KLUE 데이터셋에 대해 RoBERTa-large보다 성능이 좋다고 한다.

**KoAlpaca**  
위 모델들과 다르게 Decoder 모델도 하나 넣고 싶어서 넣어봄여. KoAlpaca는 한국어 데이터셋을 사용해서 Alpaca 학습법을 적용한 모델이다. LLaMA는 GPT-3와 유사한 구조의 사전 학습된 모델이고, 7B, 13B, 33B, 65B 등 여러 버전이 있고, 파라미터가 많을 수록 성능이 좋다고 한다. 이 LLaMA를 Instruction-following 데이터로 fine-tuning한 모델이 Alpaca이다. 따라서 KoAlpaca는 LlaMA나 Polyglot-ko(KoAlpaca의 백본)를 instruction tuning 방식으로 fine-tuning한 모델이다.  
Decoder 기반 모델이지만 출력을 잘 조정한다면 STS에도 적용해볼 수 있을 것인데.. 오버 엔지니어링인가 싶어 보류.  

<br>

> **출처**  
[KLUE Hugging Face](https://huggingface.co/datasets/klue/klue)  
[KLUE github](https://github.com/KLUE-benchmark/KLUE?tab=readme-ov-file)  
[KLUE paper](https://arxiv.org/pdf/2105.09680)  
[XLM-R 설명](https://ariz1623.tistory.com/309)  
[KoELECTRA github](https://github.com/monologg/KoELECTRA)  
[KF-DeBERTa github](https://github.com/kakaobank/KF-DeBERTa?tab=readme-ov-file)  
[KoAlpaca 설명](https://4n3mone.tistory.com/8)  