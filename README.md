# CurtainCall

![result](https://github.com/oeccsy/CurtainCall/assets/77562357/4129b579-eed0-487a-975f-cd500de78864)

## 기획의도
> ### 막이내린 기억. 다시 떠올리기 위한 행위.  
```
깜깜한 모습으로 시작한 화면은 유저가 손짓하면 감춰진 모습이 드러납니다.
유저가 프로그램을 시작할 때 지정한 이미지 입니다.

하지만 그 모습도 조금이 지나면 다시 깜깜한 모습으로 돌아갑니다.

유저는 사진속 그 장면을 다시 보기위해 계속 손짓하게되는
CurtainCall을 인터렉티브 미디어 아트로 구현한 작품입니다.
```

## 실행방법
- 실행에 `웹캠`이 필요합니다.
- `/Resources/MainImage/` 경로의 이미지를 원하는 이미지로 교체합니다.
    - 해당 경로에는 파일이 단 하나 존재해야 합니다.
- `/Resources/SubImage/` 경로의 이미지를 원하는 이미지로 교체합니다.
    - 해당 경로에는 파일이 100개 이상 존재 해야 합니다.
    - 기존의 이미지를 지우고 채워 넣어야 합니다.
- `/Python/Release/media_art.py` 를 실행합니다.
    - 잠시 기다리면 시작됩니다.  

## 구현

<img src="https://github.com/oeccsy/CurtainCall/assets/77562357/67b05be3-23b3-4edc-905e-8e7b65b90009" width="30%" height="30%">  

> ### 모자이크 아트
- 수십장의 사진으로 표현된 한장의 이미지 입니다.
- 작은 이미지들이 모여 하나의 이미지를 새롭게 이뤄냅니다.  
- 총 100장의 서로다른 이미지들이 모여 하나의 이미지를 이루도록 구현하였습니다.
<br>
<br>
<img src="https://github.com/oeccsy/CurtainCall/assets/77562357/b29a40f1-c580-494b-8989-a016438d45f7" width="30%" height="30%">   

> ### 카툰  

- `cartoonize` 를 진행하여 이미지를 새롭게 표현합니다.  
- `clustering` 을 적용하여 적절한 색상으로 자연스럽게 표현되도록 구현하였습니다.  
<br>  
<br>  

> ### 인터렉티브 미디어아트  
- `pose estimation`을 적용하여 사용자의 `손`을 인식합니다.
- `검지`를 인식한 위치에서 상호작용이 이뤄지며, 이미지가 드러납니다.
- 기존의 `landmark`를 추적하는 방법으로 드러난 이미지가 `사라지는 animation` 연출을 구현하였습니다.
- 이미지는 유저의 상호작용 진행 시간에 따라 `Edge` -> `Mosaic Art` -> `Blur` -> `Cartoon` 스타일의 이미지로 나타납니다.

    - 시연 영상

https://github.com/oeccsy/CurtainCall/assets/77562357/1de65269-6dd9-4c74-9224-1c7eb1f279ff


### 비하인드
- 유니티 엔진을 이용하려 했으나 현재 구현 목표로는 굳이 사용할 이유가 없어서 추후 업데이트에서 활용할 예정입니다.
- 해당 프로젝트를 활용한 모자이크 아트는 생일 주인공에게 선물로 전달되었습니다.
