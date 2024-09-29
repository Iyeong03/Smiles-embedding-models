# Smiles-embedding-models
데이콘에서 개최된 제2회 신약개발 경진대회를 참가하면서 사용한 embedding model 정리
### GNN (Graph Neural Network)
- Message Passing을 통해 노드 간의 정보를 교환하고 집계하는 과정을 통해 임베딩을 학습
### GCN (Graph Convolutional Network)
- 그래프 구조의 데이터를 처리하는 모델로 **노드**간의 관계와 **특징**을 활용하여 다야한 그래프 기반의 학습을 수행하는 모델
- 각 노드가 자신의 이웃 노드로부터 정보를 받아들이고, 이를 기반으로 임베딩을 계산
- 인접 행렬의 정규화와 합성곱 연산을 사용하여 노드의 임베딩을 학습
- GCN 레이어를 쌓아서 그래프에서 멀리 떨어진 노드 간의 정보를 결합할 수 있음
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
### SelFormer
- SELFIES를 활용하여 분자 구조 정보를 학습하는 모델로 self-supervised learning 방식의 모델
- RoBERTa Tokenizer를 활용하여 Smiles를 tokenize
- 제작자가 제공한 SMILES 데이터들을 이용하여 SelFormer pre-train 진행 (각기 다른 파라미터로 3개의 모델 pre-train)
- loss가 낮은 모델 기준으로 대회 데이터를 이용하여 embedding 진행
