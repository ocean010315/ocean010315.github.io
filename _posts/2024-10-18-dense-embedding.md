---
title: Retrieval 구현(SBERT를 곁들인...)
date: 2024-10-18 00:00:00 +09:00
categories: [NLP, 구현]
tags: [NLP, MRC]
use_math: true
---

### 배경
MRC 프로젝트 수행 시 질문이 입력되면 그에 해당하는 문서를 찾기 위해 Dense Embedding을 구현하고자 한다.  
제공받은 데이터셋의 경우, 문서를 보고 만든 질문들이 많아 키워드를 사용하여 문서를 탐색하는 방식인 Sparse Embedding이 보다 적합하다고 하지만, 개인적인 공부를 위해, 그리고 실험해보며 직접 경험해보기 위해 Dense Embedding 방식 또한 시도하였다.  

마침 embedding을 보다 잘 수행할 수 있도록 학습된 SBERT를 보다 쉽게 사용할 수 있는 Sentence Transformer라는 API가 있길래 해당 API를 가져다 사용했다.  
(전체 코드는 public 전환 후 링크 첨부 예정)  

### Bi-Encoder와 Cross Encoder
Sentence Transformer의 문서를 참고하면([여기](https://sbert.net/examples/applications/retrieve_rerank/README.html)) Retrieve & Re-Rank에 대한 글이 있다.  
일반적으로 Retrieval 자체는 두 단계로 수행하는 것이 성능이 좋고, Sentence Transformer 모듈을 사용하여 각 단계에서 Bi-Encoder와 Cross Encoder를 사용할 수 있다. 일반적으로 Bi-Encoder를 사용해 Retrieval을 수행하고, Cross Encoder를 사용해 탐색한 문서에 대해서 Re-Ranking을 수행하여 최종적으로 출력한다.  

### Bi-Encoder  
임베딩을 수행하기 위해 학습할 때 질문과 문서를 서로 다른 모델에 입력하는 방식이다. 서로 다른 모델이라는 것은 구조가 완전히 다른 두 개의 모델일 수도 있지만, 데이터의 입출력 형태와 모델 각각의 특성을 유지하기 위해 구조가 같은 모델을 개별적으로 학습한다는 뜻이기도 하다.  
질문과 지문을 개별적으로 임베딩 한 후 출력된 두 개의 벡터를 사용해 유사성을 계산한 뒤, 손실 함수를 사용하여 역전파를 수행한다.  
질문과 문서를 서로 다른 모델로 학습하기 때문에 상호 간의 특성을 파악하는 데 어려운 경향이 있으나, 질문과 지문을 매칭하는 데 필요한 리소스를 줄여준다는 장점이 있다.
이때 Sentence Transformer에서는 전용 Trainer와 임베딩 벡터의 유사성을 고려한 다양한 종류의 Loss, Evaluation 클래스들을 제공한다. 여기서 전부 소개하고 싶지만 종류가 많아 다른 포스팅으로 소개하고자 한다.  

**`TraininingArguments`**  
[Sentence Transformer API](https://sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentencetransformertrainingarguments) 참고  
학습에 필요한 파라미터 지정  
```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformersTrainer

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') # Sentence Transformer를 지원하는 모델이어야 한다.

args = SentenceTransformerTrainingArguments(
    # 필수
    output_dir=output_dir,
    # 옵션: 하이퍼파라미터 조정
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    weight_decay=0.001,
    warmup_ratio=0.01,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # 일부 loss에서 설정해야 하는 경우가 존재
    # 중간 점검을 위한 옵션들
    eval_strategy="steps", # 지정한 step마다 검증
    eval_steps=100,
    save_strategy="steps", # 지정한 step에서 중간 저장
    save_steps=500,
    save_total_limit=2, # 총 저장 개수
    logging_steps=100
    run_name="sbert-5",  # 'wandb가 설치되어 있다면 자동으로 사용'
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=eval_datasets,
    loss=loss
)
```

**`Loss`**  
[API 참고](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#)  
데이터셋의 특징에 따라 지정할 수 있는 loss가 다르기 때문에 추후 기회가 된다면 정리할 예정이다.  
현재 사용하는 데이터셋에는 질문과 지문밖에 없었기 때문에, `MegaBatchMarginLoss`를 사용했다. 이 Loss는 배치 내에서 코사인 유사도, 내적 등을 사용해 가장 유사하지만 정답이 아닌 경우를 골라 Negative Sampling을 자동으로 수행해주는 Loss이다.  
```python
from sentence_transformers.losses import MegaBatchMarginLoss

loss = MegaBatchMarginLoss(model=self.model)
```

**`Trainer`**  
[API 참고](https://sbert.net/docs/package_reference/sentence_transformer/trainer.html)  
```python
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=valid_datasets,
    loss=loss
)

trainer.train()
```  

### Cross Encoder
Cross Encoder는 Bi-Encoder의 문제를 해결하기 위해 질문과 지문을 연결하여 하나의 sequence 데이터로 만든다.  
모든 질문에 대해서 지문을 매칭해야 하기 때문에 인력과 리소스가 소모되지만, Bi-Encoder보다 성능이 좋다고 알려져 있다.  
Sentence Transformer에서 이를 보다 편리하게 구현하도록 하는 Cross Encoder 모듈이 있지만, Sentence Transformer의 다른 모듈들에 비해 업데이트가 느리다. 즉, Trainer를 사용할 수 없다..ㅠ  
개별 모듈에 대한 설명도 많지 않기 때문에 아래 API를 참고하고, Github 코드를 참고하는 것이 좋을 수 있다.  
[API](https://sbert.net/docs/cross_encoder/training_overview.html), [Github](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py)  

**`Dataset & Model`**  
Bi-Encoder를 사용할 때는 loss, evaluator에 따라 데이터셋을 설정하는 방식도 조금 달랐지만, Cross Encoder 모듈을 사용할 때는 InputExample와 pytorch의 DataLoader를 사용한다.  
```python
from sentence_transformers import CrossEncoder, InputExample  

train_reranker = []
for i in tqdm(range(train.num_rows), desc='pre-train'):
    train_reranker.append(InputExample(texts=[train['question'][i], train['context'][i]], label=1)) # texts를 question, context 순서로 맞춰주고, 두 시퀀스는 동일한 의미이기 때문에 label로 유사도가 같다는 뜻의 1을 부여한다.
train_dataloader = DataLoader(train_reranker, shuffle=True, batch_size=batch_size)

model = CrossEncoder('klue/roberta-base') # AutoModel로 불러올 수 있는 모델이면 어떤 것이든 가능하다.
```  

**`Evaluator`** - 선택 사항  
별도로 제공되는 Loss가 없기 때문에 자동으로 지정되는 손실 함수를 사용했다.  
학습 도중이나 학습 완료 후 결과를 확인할 수 있도록 evaluator를 지정한다. 이때 학습 도중 이 evaluator를 사용하면 해당 배치 내에서의 정확도를 분석해준다. 학습 완료 후 전체 문서를 대상으로 evaluator를 수행하면 너무 오래 걸리는 경향이 있다.  
```python
from from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

evaluator = CERerankingEvaluator(samples=valid_reranker, at_k=10, name='ranker')
```  

**`Train`**  
데이터셋과 evaluator를 선언했다면 학습을 수행하면 된다. 이때 Trainer를 사용하지 않는다.  
```python
self.model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    output_path=output_dir,    
    weight_decay=0.001,
    evaluator=evaluator, # 옵션
    evaluation_steps=500,
    save_best_model=True
)
```

### 결과
초반에 언급했던 것처럼, 제공받은 데이터셋은 문서를 보고 질문을 만든 경향이 커 질문과 문서의 의미를 통해 Retrieval을 수행하기 어렵다. 따라서 좋은 성능은 확인하지 못했다.  
하지만 Retrieval의 두 단계 구성, Bi-Encoder와 Cross Encoder를 구현해봄으로써 Dense Embedding 방식을 이해할 수 있어 좋았다.  
정확히 그 수치와 탐색한 문서들의 특징을 분석한 결과가 있으면 좋았겠지만, 시간에 쫓겨 분석을 제대로 수행하지 못했던 점 또한 아쉽다.