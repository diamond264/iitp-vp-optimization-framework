# 온디바이스 보이스피싱 탐지를 위한 인공신경망 경량화 프레임워크

한국어 보이스피싱 탐지 모델에 **양자화, 구조적 가지치기, 지식증류**를 적용하고 탐지 성능과 서버 GPU 추론 시간을 비교하는 연구용 소프트웨어입니다. 대형 언어모델의 예측 확률을 활용하여 DistilKoBERT 기반 학생 분류기를 학습합니다.

**이 저장소는 코드만 공개합니다.** HyperCLOVA를 포함한 모델 파일, 토크나이저 자산, 학습·검증·테스트 데이터, 소프트 라벨, 체크포인트, 실험 로그와 결과 파일은 포함하지 않습니다. 원본 비공개 저장소의 Git 이력도 포함하지 않습니다. 실행에는 사용 권한을 확보한 데이터와 교사 모델이 별도로 필요합니다.

## 주요 기능

- 교사 모델의 정상·피싱 출력 확률 추출 및 소프트 라벨 생성
- DistilKoBERT 인코더와 이진 분류 헤드를 이용한 학생 모델 학습
- 지식증류와 정답 라벨 단독 학습 비교
- INT8 및 NF4 4비트 양자화 비교
- Transformer 레이어 제거 비율 10·30·50·80%의 구조적 가지치기 비교
- 정확도, 정밀도, 재현율, F1-score, AUC 계산
- CUDA event 기반 모델 forward 실행 시간 측정

본 공개 범위에는 Android 앱, 음성인식 모델, 동적 모델 갱신 알고리즘이 포함되지 않습니다.

## 프로젝트 구조

```text
iitp-vp-optimization-framework/
├── configs/
│   ├── teacher_labels.yaml         # 교사 소프트 라벨 추출 설정
│   ├── distil_default.yaml         # 지식증류 학습 설정
│   └── distil_hard_only.yaml       # 정답 라벨 단독 학습 설정
├── scripts/
│   ├── extract_teacher_labels.sh   # 소프트 라벨 추출 실행
│   ├── run_distill.sh              # 학생 모델 학습 실행
│   └── audit_public_release.py     # 공개 파일 및 Git 이력 점검
├── src/distil_vp/
│   ├── teacher_labels.py          # 교사 모델 예측 확률 추출
│   ├── model.py                   # 학생 모델 구조 정의 (가중치 아님)
│   ├── dataset.py                 # 데이터 로더 코드 (데이터 아님)
│   ├── train.py                   # KL 및 CE 결합 학습
│   ├── evaluate.py                # 이진 분류 평가 지표
│   ├── compare.py                 # 학생 모델 학습 조건 비교
│   ├── llm_compare.py             # 양자화 및 가지치기 비교
│   └── latency_bench.py           # 서버 GPU 추론 시간 비교
├── requirements.txt
└── .gitignore
```

## 실행 환경

Linux, Python 3.11, CUDA 지원 NVIDIA GPU를 기준으로 구성한 코드입니다. 양자화 및 GPU 시간 측정은 호환되는 CUDA·bitsandbytes 환경이 필요합니다. 모델 크기와 입력 길이에 맞는 GPU 메모리를 확보해야 합니다.

```bash
git clone https://github.com/diamond264/iitp-vp-optimization-framework.git
cd iitp-vp-optimization-framework
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

저장소 최상위 폴더에서 명령을 실행합니다. 의존성 파일은 전체 lock 파일이 아니므로 실제 사용한 패키지 버전을 별도로 기록해야 합니다. 이번 코드 공개를 위해 학습이나 성능 실험을 재실행하지는 않았습니다.

## 비공개 입력 준비

다음 경로는 사용자가 로컬에서 준비하는 경로이며 저장소에 포함되지 않습니다.

| 입력 | 로컬 경로 예시 | 요구 사항 |
|---|---|---|
| 교사 모델 | `private_models/teacher/` | Transformers에서 읽을 수 있는 모델 및 토크나이저 |
| 학습 데이터 | `private_data/train.csv` | `text`, `label` 열 |
| 테스트 데이터 | `private_data/test.csv` | `text`, `label` 열; 학습 데이터와 분리 |
| 소프트 라벨 데이터 | `private_data/train_teacher_labels.csv` | 위 열과 `teacher_p` 열; 아래 명령으로 생성 |

`label`은 정상 0, 피싱 1입니다. `teacher_p`는 교사 모델의 피싱 확률입니다. 개인정보 및 이용 허가가 없는 자료를 저장소나 이슈에 업로드하지 마세요.

## 실행 방법

### 1. 교사 소프트 라벨 생성

`configs/teacher_labels.yaml`의 모델·데이터 경로를 확인한 뒤 실행합니다.

```bash
bash scripts/extract_teacher_labels.sh
```

### 2. 지식증류 및 비교군 학습

```bash
bash scripts/run_distill.sh configs/distil_default.yaml
bash scripts/run_distill.sh configs/distil_hard_only.yaml
```

기본 KD 설정은 α=0.7, 학생 온도 T=2.0, 교사 확률 추출 온도 1.0입니다. 손실은 `α × T² × KL(교사 확률 || 학생 확률) + (1−α) × CE`입니다. 정답 라벨 단독 조건은 α=0입니다. 검증 F1 기준 최적 모델과 학습 이력은 로컬 `outputs/`에만 저장합니다.

### 3. 학생 모델 성능 비교

```bash
python -m distil_vp.compare \
  --kd_ckpt outputs/distil_run/best_student.pt \
  --hard_ckpt outputs/hard_only_run/best_student.pt \
  --test_csv private_data/test.csv \
  --out_json outputs/comparison_test.json
```

### 4. 양자화 및 가지치기 비교

```bash
python -m distil_vp.llm_compare \
  --model_name private_models/teacher \
  --test_csv private_data/test.csv \
  --settings original int8 int4 prune_10 prune_30 prune_50 prune_80 \
  --out_json outputs/llm_comparison.json
```

### 5. 서버 GPU 추론 시간 측정

```bash
python -m distil_vp.latency_bench \
  --model_name private_models/teacher \
  --test_csv private_data/test.csv \
  --kd_ckpt outputs/distil_run/best_student.pt \
  --hard_ckpt outputs/hard_only_run/best_student.pt \
  --out_json outputs/latency.json
```

예비 입력 5건과 본 측정 100건을 위해 최소 105건의 입력을 준비합니다. 토큰화와 GPU 입력 전송은 측정 전에 수행합니다. 여기서 얻는 모델 forward 시간은 전체 서비스 지연이나 스마트폰 공인시험 xRT와 다릅니다.

## 구현 범위 및 유의사항

- 교사 출력 토큰 ID 15·16은 기존 HyperCLOVA-X 실험 설정입니다. 다른 모델이나 토크나이저에 그대로 적용하지 말고 정상·피싱 답변 토큰을 확인하여 코드를 수정해야 합니다.
- 가지치기는 `model.model.layers` 구조를 전제로 하며 제거 후 추가 미세조정을 포함하지 않습니다. 모든 모델에 호환되는 범용 가지치기 구현은 아닙니다.
- `student.hf_id`는 기본 토크나이저 설정입니다. 학생 인코더는 `model.py`의 `STUDENT_HF_ID`를 사용하므로 다른 모델로 변경할 때 두 설정을 함께 검토해야 합니다.
- 학생 사전학습 모델은 실행 시 Hugging Face에서 내려받을 수 있지만, 파일 자체를 이 저장소에 배포하지 않습니다. `trust_remote_code=True`가 있는 로딩 코드는 신뢰하는 모델 저장소에만 사용하세요.
- `.pt` 파일은 신뢰할 수 있는 출처의 체크포인트만 사용하세요. 결과 재현에는 동일한 모델·데이터·토크나이저·패키지·하드웨어 조건이 필요합니다.
- 전체 세트 실행 시간(`elapsed_s`)과 CUDA event 기반 추론 시간은 서로 다른 지표입니다. 양자화된 텐서의 `numel()` 값은 원래 논리적 파라미터 수와 다를 수 있습니다.
- 교사 확률 누락 시 KD 항은 해당 입력에서 제외됩니다. 전체 교사 확률이 누락되면 손실은 `(1−α) × CE`이므로, 정답 라벨 단독 학습은 반드시 α=0으로 설정합니다.

## 공개 전 점검

```bash
python scripts/audit_public_release.py
git status --short
```

점검 스크립트는 허용된 코드 파일만 존재하는지, 개인 절대경로·대표적인 비밀키 형식·대형 파일이 없는지 확인하고 Git 저장소가 있으면 전체 커밋의 파일도 검사합니다. 자동 검사는 보조 수단이며, 공개 전 최종 diff를 직접 확인해야 합니다. `.gitignore`는 이미 커밋된 파일이나 과거 Git 이력을 제거하지 않습니다.

## 이용 조건

모델과 데이터는 포함하지 않으며, 이용자가 각 제공자의 사용 조건과 배포 권한을 확인해야 합니다. 소스코드의 별도 라이선스는 권리자 확인 후 명시할 예정이며, 현재 공개가 제3자 모델·데이터의 재배포 허가를 의미하지 않습니다.
