# YOLO_ViTPose_Fine_Tuning
## YOLOv8과 YOLOv9,v10을 Fine tuning 해보면서 이미지 객체 감지(Image Detection)의 개념을 이해
데이터 전처리의 중요성: 학습 데이터셋을 준비하는 과정에서, 정확한 레이블링과 데이터 증강이 모델의 성능에 큰 영향을 미친다는 것을 경험
하이퍼파라미터 튜닝의 영향: 배치 사이즈, 학습률 등의 하이퍼파라미터를 조정하면서, 이들이 모델의 학습 속도와 정확도에 미치는 영향을 직접 확인
실시간 객체 감지: 자율주행, 보안 시스템 등 다양한 실제 응용 분야에서 중요한한 요소

## ViTPose를 사용해보면서 키포인트 감지(Keypoint Detection)에 대한 이해
트랜스포머 구조의 이해: ViTPose가 비전 트랜스포머(Vision Transformer) 구조를 사용한다는 점에서, Transformer 아키텍처에 대한 이해 심층화 
포즈 추정의 응용: 사람의 자세를 정확히 추정하는 것이 스포츠 분석, 의료 진단, 증강현실 등 다양한 분야에서 활용될 수 있음

### 이번 공부를 통해 과연 이 기술들을 가지고 어떤 분야에 기여할 수 있는지 더욱 고민해보고 컴퓨터 비전 분야에 대해 좀 더 이해할 수 있게되었습니다.
### 그리고 라벨링에도 바운딩박스, 폴리곤, 키포인트, 폴리라인, 시맨틱 세그멘테이션 등 직접 해보며 다양한 라벨링 기법들에 대해 이해하게 되었습니다.


출처
https://github.com/ViTAE-Transformer/ViTPose  
https://github.com/JunkyByte/easy_ViTPose  
https://github.com/ultralytics/ultralytics
https://github.com/THU-MIG/yolov10
