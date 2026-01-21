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

## 설치

### 필수 요구사항

- Python 3.7 이상
- NumPy
- Matplotlib
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

1. **평형 탄성계수 (E₀)**: 초기 평형 상태의 탄성계수를 설정합니다.

2. **주파수 범위**: 그래프에 표시할 주파수 범위를 log₁₀ 스케일로 설정합니다.
   - 예: -10 ~ 20은 10⁻¹⁰ Hz ~ 10²⁰ Hz를 의미

3. **Maxwell 요소 파라미터**: 각 Maxwell 요소의 파라미터를 조절합니다.
   - `Eᵢ (MPa)`: i번째 요소의 탄성계수
   - `τᵢ (s)`: i번째 요소의 완화시간

4. **그래프 업데이트**: 파라미터 변경 후 버튼을 클릭하여 그래프를 업데이트합니다.

5. **초기값으로 리셋**: 모든 파라미터를 초기값으로 되돌립니다.

### 초기값 (Default Parameters)

- **E₀** = 10.0 MPa
- **Maxwell 요소 1**: E₁ = 1000 MPa, τ₁ = 10⁻⁶ s
- **Maxwell 요소 2**: E₂ = 5000 MPa, τ₂ = 10⁻⁴ s
- **Maxwell 요소 3**: E₃ = 10000 MPa, τ₃ = 10⁻² s
- **Maxwell 요소 4**: E₄ = 15000 MPa, τ₄ = 1 s

## 프로그램 특징

- ✅ Generalized Maxwell 모델 구현
- ✅ 실시간 파라미터 조절
- ✅ 저장탄성계수(E') 및 손실탄성계수(E") 동시 표시
- ✅ log-log 스케일 그래프
- ✅ 사용자 친화적 GUI 인터페이스
- ✅ 4개의 Maxwell 요소 지원

## 물리적 의미

### 저장탄성계수 (E' - Storage Modulus)
- 재료가 에너지를 저장하는 능력
- 탄성 거동과 관련
- 붉은색 곡선으로 표시

### 손실탄성계수 (E" - Loss Modulus)
- 재료가 에너지를 소산하는 능력
- 점성 거동과 관련
- 녹색 곡선으로 표시

## 프로젝트 구조

```
kelvin-voight/
├── viscoelastic_modeler.py  # 메인 GUI 프로그램
├── requirements.txt         # Python 패키지 의존성
└── README.md               # 이 파일
```

## 참고문헌

- Generalized Maxwell Model for viscoelastic materials
- Ferry, J.D., "Viscoelastic Properties of Polymers"

## 라이센스

MIT License

## 작성자

Claude Code - Viscoelastic Material Modeling Tool
