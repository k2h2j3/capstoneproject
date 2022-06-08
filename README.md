# capstoneproject

CNN 머신러닝을 이용하여 신호 표지판을 분류하는 모델을 만든 후 open cv를 통해 스마트폰 카메라로 실시간으로 감지하는 어플리케이션을 개발할 목적으로 만들어짐.

1. 개발 환경

jupyter notebook, keras, sklearn, open cv



2. 개발 진행도

 CNN 머신러닝 모델링 -> accuracy 95% loss 10% 

 open cv를 통해 모델 테스트 -> 노트북 캠 사용

 스마트폰 카메라 연동 -> 스마트폰 연동 o, detection 기능 필요(진행중)

 어플리케이션 개발 -> 


3. 가중치 모델

my_model -> 10개 종류(speed classfication x)

my_model2 -> 최신 정확도(accuracy 높음)

TrafficSignModel2.ipynb(My_Neww_Model) --> detect 기능 넣기위해 새로 편집한 모델(원형 표지판으로만 학습)

4. 그외 파일

DetectingTrafficSign => 표지판 분류 기능에서 표지판 모양 인식하는 기능 추가(아직 미완성)
