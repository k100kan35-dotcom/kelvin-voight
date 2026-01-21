# 고무 컴파운드 점탄성 모델링 (Viscoelastic Rubber Compound Modeler)

고무 및 트레드 컴파운드의 진동수 의존 점탄성 거동을 모델링하는 GUI 프로그램입니다.

## 개요

이 프로그램은 **Generalized Maxwell Model**을 사용하여 고무 재료의 복소 탄성계수를 계산하고 시각화합니다.

### 모델 방정식

복소 탄성계수:
```
E(ω) = E' + iE"
```

저장탄성계수 (Storage Modulus):
```
E'(ω) = E₀ + Σ[Eᵢ·ω²τᵢ²/(1 + ω²τᵢ²)]
```

손실탄성계수 (Loss Modulus):
```
E"(ω) = Σ[Eᵢ·ωτᵢ/(1 + ω²τᵢ²)]
```

여기서:
- `E₀`: 평형 탄성계수 (Equilibrium modulus)
- `Eᵢ`: i번째 Maxwell 요소의 탄성계수
- `τᵢ`: i번째 완화시간 (Relaxation time)
- `ω`: 각진동수 (Angular frequency) = 2πf

## 주요 기능 (New Features!)

### 1. 동적 Maxwell 요소 관리
- ✅ **1~20개 Maxwell 요소 지원** - 스핀박스로 요소 개수 조절
- ✅ 실시간 요소 추가/삭제
- ✅ 각 요소별 독립적인 E와 τ 파라미터 조절

### 2. Generalized Maxwell 모델 다이어그램
- ✅ **실시간 모델 구조 시각화**
- ✅ 스프링과 댐퍼(dashpot)로 구성된 물리적 모델 표현
- ✅ E₀ 스프링 + 병렬 Maxwell 요소들 시각적 표현

### 3. 마스터 커브 데이터 피팅
- ✅ **CSV 파일 로드 기능** - 실험 데이터 불러오기
- ✅ **자동 파라미터 피팅** - Differential Evolution 알고리즘 사용
- ✅ 실험 데이터와 모델 예측 비교 시각화
- ✅ 데이터 형식: `log10(f), log10(E'), log10(E")`

### 4. 한글 폰트 지원
- ✅ 한글 깨짐 문제 해결
- ✅ 자동 폰트 감지 및 설정

## 설치

### 필수 요구사항

- Python 3.7 이상
- NumPy
- Matplotlib
- SciPy (피팅 기능용)
- Tkinter (대부분의 Python 설치에 기본 포함)

### 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 프로그램 실행

```bash
python viscoelastic_modeler.py
```

또는:

```bash
chmod +x viscoelastic_modeler.py
./viscoelastic_modeler.py
```

### GUI 사용 방법

#### 기본 파라미터 설정

1. **평형 탄성계수 (E₀)**: 초기 평형 상태의 탄성계수를 설정합니다.

2. **주파수 범위**: 그래프에 표시할 주파수 범위를 log₁₀ 스케일로 설정합니다.
   - 예: -10 ~ 20은 10⁻¹⁰ Hz ~ 10²⁰ Hz를 의미

3. **Maxwell 요소 개수**:
   - 스핀박스에서 1~20 사이 값 선택
   - "Apply" 버튼 클릭하여 요소 개수 변경
   - 다이어그램이 자동으로 업데이트됨

4. **Maxwell 요소 파라미터**: 각 Maxwell 요소의 파라미터를 조절합니다.
   - `Eᵢ (MPa)`: i번째 요소의 탄성계수
   - `τᵢ (s)`: i번째 요소의 완화시간

#### 마스터 커브 피팅 기능

1. **데이터 준비**:
   - CSV 파일 형식: 헤더 1줄 + 데이터
   - 컬럼: `log10(f), log10(E'), log10(E")`
   - 샘플 데이터 생성:
     ```bash
     python generate_sample_data.py
     ```

2. **데이터 로드**:
   - "Load Data" 버튼 클릭
   - CSV 파일 선택
   - 데이터 포인트가 그래프에 표시됨

3. **모델 피팅**:
   - "Fit Model" 버튼 클릭
   - 피팅 진행 (시간이 걸릴 수 있음)
   - 최적화된 파라미터가 자동으로 GUI에 반영됨

4. **결과 확인**:
   - 붉은색 실선: 예측된 E'
   - 녹색 실선: 예측된 E"
   - 붉은색 점: 실험 E' 데이터
   - 녹색 사각형: 실험 E" 데이터

#### 버튼 기능

- **Update Plots**: 파라미터 변경 후 그래프 업데이트
- **Reset**: 모든 파라미터를 초기값으로 되돌림

### 초기값 (Default Parameters)

- **E₀** = 10.0 MPa
- **Maxwell 요소 1**: E₁ = 1000 MPa, τ₁ = 10⁻⁶ s
- **Maxwell 요소 2**: E₂ = 5000 MPa, τ₂ = 10⁻⁴ s
- **Maxwell 요소 3**: E₃ = 10000 MPa, τ₃ = 10⁻² s
- **Maxwell 요소 4**: E₄ = 15000 MPa, τ₄ = 1 s

## 프로그램 특징

### 모델링 기능
- ✅ Generalized Maxwell 모델 구현
- ✅ 실시간 파라미터 조절
- ✅ 저장탄성계수(E') 및 손실탄성계수(E") 동시 표시
- ✅ log-log 스케일 그래프
- ✅ 1-20개 Maxwell 요소 동적 관리

### 시각화 기능
- ✅ 실시간 Maxwell 모델 다이어그램
- ✅ 스프링-댐퍼 물리적 표현
- ✅ 복소 탄성계수 그래프
- ✅ 마스터 커브 데이터 오버레이

### 데이터 분석 기능
- ✅ CSV 파일 로드
- ✅ 자동 파라미터 피팅 (Differential Evolution)
- ✅ 실험 데이터와 모델 비교

## 물리적 의미

### 저장탄성계수 (E' - Storage Modulus)
- 재료가 에너지를 저장하는 능력
- 탄성 거동과 관련
- 붉은색 곡선으로 표시

### 손실탄성계수 (E" - Loss Modulus)
- 재료가 에너지를 소산하는 능력
- 점성 거동과 관련
- 녹색 곡선으로 표시

### Generalized Maxwell Model
- E₀: 평형 상태에서의 탄성계수
- 각 Maxwell 요소: 스프링(E)과 댐퍼(τ)의 직렬 연결
- 모든 요소는 E₀와 병렬로 연결됨

## 프로젝트 구조

```
kelvin-voight/
├── viscoelastic_modeler.py      # 메인 GUI 프로그램
├── generate_sample_data.py      # 샘플 데이터 생성 스크립트
├── sample_master_curve.csv      # 샘플 마스터 커브 데이터
├── test_model.py                # 모델 검증 테스트
├── requirements.txt             # Python 패키지 의존성
└── README.md                    # 이 파일
```

## 데이터 파일 형식

마스터 커브 CSV 파일은 다음 형식을 따라야 합니다:

```csv
log10_frequency_Hz,log10_storage_modulus_MPa,log10_loss_modulus_MPa
-8.0,1.234,0.567
-7.5,1.456,0.789
...
```

- 첫 번째 컬럼: log₁₀(주파수) [Hz]
- 두 번째 컬럼: log₁₀(저장탄성계수) [MPa]
- 세 번째 컬럼: log₁₀(손실탄성계수) [MPa]

## 사용 예시

### 1. 기본 모델링

```bash
# 프로그램 실행
python viscoelastic_modeler.py

# GUI에서:
# 1. Maxwell 요소 개수 조절 (예: 6개)
# 2. 각 요소의 E와 τ 값 입력
# 3. "Update Plots" 클릭
# 4. 상단에서 모델 다이어그램 확인
# 5. 하단에서 복소 탄성계수 그래프 확인
```

### 2. 마스터 커브 피팅

```bash
# 샘플 데이터 생성
python generate_sample_data.py

# 프로그램 실행
python viscoelastic_modeler.py

# GUI에서:
# 1. "Load Data" 클릭 → sample_master_curve.csv 선택
# 2. 데이터 포인트가 그래프에 표시됨
# 3. Maxwell 요소 개수 설정 (예: 4개)
# 4. "Fit Model" 클릭
# 5. 피팅 완료 후 최적 파라미터 확인
```

## 알고리즘

### 피팅 알고리즘: Differential Evolution

- 전역 최적화 알고리즘
- 모든 파라미터(E₀, Eᵢ, τᵢ)를 동시에 최적화
- 목적 함수: log 스케일에서 E'와 E"의 MSE
- 장점: 로컬 최적해에 빠지지 않음

## 성능 팁

- **요소 개수**: 3-6개가 대부분의 경우 충분
- **피팅 속도**: 요소 개수가 많을수록 피팅 시간 증가
- **데이터 품질**: 노이즈가 적고 넓은 주파수 범위의 데이터일수록 좋은 결과

## 문제 해결

### 한글이 깨져 보일 때
- 시스템에 한글 폰트 설치 확인
- Ubuntu: `sudo apt-get install fonts-nanum`
- macOS: 기본 설치된 AppleGothic 사용
- Windows: Malgun Gothic 자동 사용

### 피팅이 수렴하지 않을 때
- Maxwell 요소 개수 조절
- 초기 파라미터 범위 확인
- 데이터 품질 점검

## 참고문헌

- Ferry, J.D., "Viscoelastic Properties of Polymers", 3rd ed., Wiley, 1980
- Lakes, R.S., "Viscoelastic Materials", Cambridge University Press, 2009
- Tschoegl, N.W., "The Phenomenological Theory of Linear Viscoelastic Behavior", Springer, 1989

## 라이센스

MIT License

## 작성자

Claude Code - Advanced Viscoelastic Material Modeling Tool

## 버전 히스토리

### v2.0 (Current)
- ✅ 동적 Maxwell 요소 관리 (1-20개)
- ✅ 실시간 모델 다이어그램 시각화
- ✅ 마스터 커브 데이터 피팅 기능
- ✅ 한글 폰트 지원

### v1.0
- ✅ 기본 Generalized Maxwell 모델 구현
- ✅ 4개 고정 Maxwell 요소
- ✅ 기본 GUI
