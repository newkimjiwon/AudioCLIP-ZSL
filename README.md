# ESC-50 음향 분류 모델

이 프로젝트는 ESC-50 데이터셋을 사용하여 주변 소리를 50가지 카테고리로 분류하는 딥러닝 모델입니다.

## 프로젝트 설명

<이 프로젝트를 시작하게 된 계기, 목표, 사용한 모델 아키텍처(예: CNN, ResNet 등)에 대해 간략하게 설명해 주세요.>

## 주요 기능

-   ESC-50 데이터셋 전처리
-   PyTorch 기반의 음향 분류 모델 학습
-   학습된 모델을 사용한 음향 파일 분류(추론)

## 설치 방법

1.  **저장소 복제**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-project-name>.git
    cd <your-project-name>
    ```

2.  **필요한 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

3.  **데이터셋 준비**
    ESC-50 데이터셋을 다운로드하여 `data/audio/` 폴더에 압축을 해제하세요.
    (ESC-50 공식 링크: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50))

    최종적으로 아래와 같은 구조가 되어야 합니다.
    ```
    data/
    ├── audio/
    │   ├── 1-100032-A-0.wav
    │   ├── 1-100038-A-14.wav
    │   └── ...
    ├── esc50.csv
    └── data_split.json
    ```

## 사용 방법

### **1. 데이터 전처리**

<만약 전처리 스크립트를 따로 실행해야 한다면, 여기에 명령어를 적어주세요. 예시:>
```bash
python src/preprocess.py
```

### **2. 모델 학습**

<학습을 시작하는 명령어를 여기에 적어주세요. 예시:>
```bash
python src/train.py --config config.json
```

### **3. 추론 (Inference)**

<학습된 모델로 새로운 오디오 파일을 분류하는 방법을 설명합니다. 예시:>
```bash
python src/inference.py --model_path <모델_가중치_파일_경로> --audio_file <분류할_오디오_파일>
```

## 파일 설명

-   **`requirements.txt`**: 프로젝트 실행에 필요한 Python 패키지 목록
-   **`config.json`**: 학습 및 모델 파라미터 설정 파일
-   **`data/`**: 데이터셋 메타데이터 및 정보 저장 폴더
    -   `esc50.csv`: ESC-50 데이터셋의 공식 메타데이터
    -   `data_split.json`: 학습/검증/테스트 데이터 분할 정보
-   **`src/`**: 주요 소스 코드
    -   `preprocess.py`: 오디오 데이터를 Mel-spectrogram 등으로 변환하는 전처리 스크립트
    -   `train.py`: 모델을 정의하고 학습시키는 스크립트
    -   `inference.py`: 학습된 모델로 새로운 데이터를 예측하는 스크립트