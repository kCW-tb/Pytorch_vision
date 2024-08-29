# Pytorch_vision
Pytorch 공식 사이트의 vision reference코드를 연구 목적에 맞게 수정함.
image classification, object detection, Semantic Segmentation으로 구성되어 있음.

진행한 작업
1. Labelme labeling tool에 대해서 코드와 접목시켜 적용.
2. train dataset에 대해서 데이터 증식을 적용.
3. 학습에 대해 loss값과 acc값을 tensorboard 서버에 연동해서 기록.
4. 생성된 model.pth에 대해서 onnx파일로 변환. (Segmentation code)
5. 연구실에서 사용하는 로봇에 대해 object detection 학습 및 테스트 진행.
6. 연구실 바닥을 토대로 labeling을 수행하고 학습 및 테스트.

image classfication
![image](https://github.com/user-attachments/assets/e6c8bff9-b501-433c-82b1-2fbd68237e2d)

segmantic segmentation
![image](https://github.com/user-attachments/assets/fb484b2c-451f-4a3d-bae8-3f333af9ad7f)
![image](https://github.com/user-attachments/assets/7d8488c2-84bb-41b0-a24d-6d57b7022efd)

데이터 증식 예시.
![image](https://github.com/user-attachments/assets/1be99300-e0bd-4564-958b-14dd4711548c)


Modifying Official Site Code
Source code address : https://github.com/pytorch/vision
