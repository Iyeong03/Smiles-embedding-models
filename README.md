# Smiles-embedding-models
데이콘에서 개최된 제2회 신약개발 경진대회를 참가하면서 사용한 embedding model 정리 (최종 등수 Public : 42등 score : 0.65308)
### GNN (Graph Neural Network)
###- 그래프 구조 데이터를 처리하고 학습하는 데 사용되는 신경망 모델
###- Message Passing을 통해 노드 간의 정보를 교환하고 집계하는 과정을 통해 임베딩을 학습
- 코드상에서는 GIN Convolutional Layer를 두 번 적용하고 그 결과를 global mean pooling을 통해 그래프 수준의 vector로 요약한 후, FCN을 통해 최종 예측을 진행하도록 함

- Chem의 GetAtoms 등 사용
- GCNConv, GINConv, GNNConv 등 여러 개의 layer를 사용해보았지만 큰 성과는 없었음
- 어떻게 잘 조합하느냐도 매우 중요한 듯 한데, 그래프 모델 자체를 처음 써보는 것이라 어떻게 파라미터를 조정해야 잘 나올 수 있는지 몰랐음
- 후에 VAE와 결합했을 때도 vae 모델 구현을 이상하게 해서 그런지 크게 좋은 결과가 나오지는 않음
### GCN (Graph Convolutional Network)
- 그래프 데이터에서 노드와 간선의 구조를 학습하는 데 특화된 그래프 신경망(GNN)의 한 유형
- 그래프에서 노드 간의 관계와 각 노드의 특징을 기반으로 그래프의 특성을 학습하는 데 사용
- 노드의 초기 특징을 입력받아, 그래프 구조(간선 정보)를 이용해 각 노드의 특징을 GCN 레이어를 통해 업데이트
- 그 후, Global Mean Pooling을 통해 그래프의 노드 특징을 평균화하여 하나의 벡터로 요약하고, 이를 완전 연결 레이어에 전달하여 최종 예측값을 산출
- GCN 레이어를 쌓아서 그래프에서 멀리 떨어진 노드 간의 정보를 결합할 수 있음
- 화학 구조 분석, 소셜 네트워크 분석, 분자 특성 예측 등의 작업에 적합
### GAT (Graph Attention Network)
- GCN + Attention Mechanism으로 어텐션 기반으로 그래프 구조를 학습하는 모델
- 각 이웃 노드의 중요도를 학습하는 방법을 통해 더 유연하게 정보를 통합할 수 있음
  - Attention Mechanism : 각 노드가 이웃 노드로부터 받는 정보에 가중치를 부여함으로써 중요한 이웃 노드의 정보를 더 많이 반영함
  - Self-attention weight learning : 각 edge의 attention score를 학습함으로써 그래프 내의 다양한 관계를 고려할 수 있음  
### Node2vec
- 그래프 구조의 데이터에서 효과적으로 노드 임베딩을 생성하는 대표적인 알고리즘 중 하나
- 그래프의 노드 간 관계를 효율적으로 벡터 형태로 표현
- Random Walk를 활용하여 주어진 노드로부터 그래프 위를 이동하면서 노드 시퀀스를 생성하는 방식으로 **DFS**와 **BFS**의 장점을 모두 결합해서 노드 간의 local정보와 global정보를 함께 학습
  - DFS(Depth-First Search) : 특정 노드의 깊은 관계를 탐색하며 전역적인 정보(유사성)을 반영
  - BFS(Breadth-First Search) : 가까운 이웃 노드 간의 관계를 집중적으로 탐색하며 노드 간의 지역적 유사성을 반영
- 임베딩 과정
  1. Random Walk probability 계산
  2. Graph의 각 노드 u에 대하여 길이 l만큼의 random walk 실행
  3. node2vec을 SGD로 최적화
- 기존의 DeepWalk 모델의 단점을 보완한 모델로 유연한 탐색과 다양한 그래프의 구조적 관계를 학습할 수 있다.
- 코드 상에서는 임베딩 값을 XGBoost 모델에 적용하여 최종 예측을 진행함
 ### GloVe (Global Vectors for Word Representation)
 - 정적(Static) 단어 임베딩을 생성하는 모델로 각 단어를 고정된 벡터 공간에 mapping
 - 전체 corpus의 static 정보를 활용해 단어 간의 co-occurrence information(동시 발생 정보)를 학습
 - Co-occurrence Matrix를 생성하고 이 정보를 활용해 Co-occurrence probability를 계산하여 단어간의 관계성을 파악
 - 학습된 후의 모든 단어는 항상 동일한 벡터로 표현되고 전체 코퍼스의 통계적 정보를 활용해 단어 간의 유사성을 효과적으로 표현할 수 있음
### BERT (Bidirectional Encoder Representations from Trnasformers)
- Contextual embedding을 생성하는 모델로 Transformer 구조를 기반으로 학습
- 단어를 양방향으로 동시에 고려하여 임베딩을 생성하는 데 도움이 됨
- MLM과 NSP를 목표로 pre-trained된 base 모델을 사용
- 계산 복잡도가 높아서 많은 메모리 필요로 함
### SELFormer (github HUBioDataLab/SELFormer)
- SMILES를 SELFIES로 변환하여 학습에 이용
- 제작자가 제공한 SMILES 데이터를 활용하여 임베딩 모델 pre-train (각각 파라미터를 다르게 해서 3가지의 모델을 사전학습)
- SMILES를 Tokenization하는 데에는 RoBERTa Tokenizer사용하여 분자구조의 유의미한 구조를 담도록 함
- 사전학습한 모델 중 loss가 가장 낮은 모델을 이용하여 embedding
### Embedding Class
- nn.Embedding은 정수 인덱스(원자 또는 결합)를 고정된 차원의 벡터로 매핑하는 방법
- 이 임베딩을 사용하는 이유는 정수 인덱스 자체로는 의미를 파악하기 어렵지만, 임베딩 벡터는 학습 과정을 통해 각 원자와 결합의 특징을 담게 되기 때문
- 예를 들어, GNN 모델에서 각 원자(또는 결합)의 임베딩 벡터를 입력으로 사용하여 그래프 구조를 학습함
1. **AtomEmbedding**
   - AtomEmbedding 클래스는 분자 구조에서 원자 유형을 임베딩 벡터로 변환하는 기능을 함
   -  이 클래스는 주어진 원자 인덱스를 입력받아, 각 원자에 대응하는 임베딩 벡터를 반환하여 그래프 신경망에서 원자 특징을 학습할 수 있도록 함
   -  생성자에서 nn.Embedding을 사용하여 원자 타입을 고정된 차원의 벡터로 변환할 수 있는 임베딩 테이블을 만듬
   -  forward 메서드에서 입력으로 주어진 atom 인덱스를 통해 해당 원자의 임베딩 벡터를 반환하고 이때, 이 atom 인덱스는 각 원자가 어떤 원소인지를 나타냄
2. **BondEmbedding**
   - BondEmbedding 클래스는 분자 구조에서 결합 유형을 임베딩 벡터로 변환하는 역할을 함
   - 입력으로 들어온 결합 인덱스를 통해 해당 결합에 맞는 임베딩 벡터를 반환하여, 결합의 특징을 학습할 수 있도록 함
### Fingerprint by MorganGenerator
- SMILES를 Vector로 임베딩하는 기법
### Tokenizer
### - by Tensorflow keras
- 큰 성과는 없었음
- 모듈 활용을 잘 하지 못한 것이 패착의 원인으로 짐작됨
- 결과가 단일 벡터로 나오는 것이 문제
### - by Transformer (hard coding)
-  단순하게 트랜스포머의 인코더 부분만 구현 후 실행한 것이라 그런지 효과가 좋지 않았음
-  함께 사용한 모델은 여전히 keras 모듈
### - by hugging face
- max_length에 맞춰 padding을 1로 줌
- 그 후 attention mask를 각가 줌 (padding 부분은 0으로)
### Mol2Vec
- 분자 구조를 벡터로 변환하는 임베딩 기법으로, 자연어 처리에서 사용하는 Word2Vec 알고리즘을 화학 구조에 적용한 것
- Mol2Vec임베딩 기법과 다양한 모델(RF, XGBoost, LGBM, SVM, KNN)을 조합해보았을 때 KNN모델로 학습하는 것이 가장 성능이 좋았음

# Regression models
### XGBoost
- 모듈 불러와서 사용
- 앙상블 기법 사용 (bagging)
### LGGM
- 모듈 불러와서 사용
### VAE
- 모듈 사용
- 직접 구현 -> 사이즈 문제로 실패
### Transformer
- hugging face (seyonec/PubChem10M_SMILES_BPE_180k)
- pretrained된 모델과 target 값이 다르기에 사전 학습 없이 진행함
- 그래서인지 score가 0.57에서 그침
### SELFormer
- github (HUBioDataLab/SELFormer)
- embedding된 값을 이용하여 SELFormer를 통해 Regression task를 수행하도록 함
- 모델 학습과 예측할 때 task 파라미터 값을 입력해야 하는데 제작자가 제공하는 종류는 4가지로 우리가 해야하는 task와 딱 맞는 것이 없어서 score가 0.54정도로 낮게 나온거 같음
### KNN (최종적으로 예측에 사용한 회귀모델)
- 새로운 데이터 포인트와 기존 데이터 포인트 간의 거리를 계산하여, 가장 가까운 k개의 이웃 데이터를 선택하고, 이들의 정보를 기반으로 분류 또는 회귀 작업을 수행함
- 하이퍼파라미터 k는 모델의 성능에 큰 영향을 미치며, 작은 k값은 과적합(overfitting), 큰 k값은 과소적합(underfitting)의 위험이 있음
- 모델의 성능을 위해서 grid search를 통해서 모델의 최적의 하이퍼파라미터를 찾음
- 여러 하이퍼파라미터를 설정해본 결과, n_neighbors는 8, 가중치 방식은 distance방식, 거리 측정 방식은 맨허튼 거리가 가장 성능이 좋았음
