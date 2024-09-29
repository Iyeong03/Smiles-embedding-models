# Smiles-embedding-models
데이콘에서 개최된 제2회 신약개발 경진대회를 참가하면서 사용한 embedding model 정리
### GNN (Graph Neural Network)
- Message Passing을 통해 노드 간의 정보를 교환하고 집계하는 과정을 통해 임베딩을 학습
### GCN (Graph Convolutional Network)
- 그래프 구조의 데이터를 처리하는 모델로 *노드*간의 관계와 *특징*을 활용하여 다야한 그래프 기반의 학습을 수행하는 모델
- 각 노드가 자신의 이웃 노드로부터 정보를 받아들이고, 이를 기반으로 임베딩을 계산
- 인접 행렬의 정규화와 합성곱 연산을 사용하여 노드의 임베딩을 학습
- GCN 레이어를 쌓아서 그래프에서 멀리 떨어진 노드 간의 정보를 결합할 수 있음
### GAT (Graph Attention Network)
- GCN + Attention Mechanism으로 어텐션 기반으로 그래프 구조를 학습하는 모델
- 각 이웃 노드의 중요도를 학습하는 방법을 통해 더 유연하게 정보를 통합할 수 있음
  - Attention Mechanism : 각 노드가 이웃 노드로부터 받는 정보에 가중치를 부여함으로써 중요한 이웃 노드의 정보를 더 많이 반영함
  - Self-attention weight learning : 각 edge의 attention score를 학습함으로써 그래프 내의 다양한 관계를 고려할 수 있음  
