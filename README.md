# 주식 분석 파이프라인 (Stock Analysis Pipeline)

테마 기반 종목 발굴 및 투자 판정 시스템

## 목차
- [개요](#개요)
- [파이프라인 흐름도](#파이프라인-흐름도)
- [각 단계별 상세](#각-단계별-상세)
  - [Layer 1: 데이터 수집 (Ingest)](#layer-1-데이터-수집-ingest)
  - [Layer 2: 데이터 처리 (Processing)](#layer-2-데이터-처리-processing)
  - [Layer 3: 섹터 분석 (Analysis)](#layer-3-섹터-분석-analysis)
  - [Layer 4: 필터링 (Filtering)](#layer-4-필터링-filtering)
  - [Layer 5: 점수화 (Scoring)](#layer-5-점수화-scoring)
  - [Layer 6: 최종 판정 (Decision)](#layer-6-최종-판정-decision)
- [핵심 개념](#핵심-개념)
- [설치 및 실행](#설치-및-실행)

---

## 개요

네이버 테마 크롤링 → DART/뉴스 분석 → LLM 기반 섹터 라벨링 → 수급/추세 분석 → 필터링 → 점수화 → 최종 투자 판정까지 자동화된 파이프라인입니다.

**핵심 특징:**
- **이원화 필터링**: 실적형(Track A) / 성장형(Track B) 섹터별 다른 기준 적용
- **3계층 섹터 분석**: S_Flow(수급) + S_Breadth(결속력) + S_Trend(추세)로 Tier 분류
- **LLM 통합**: Local LLM으로 재료/심리 분석
- **Decision Matrix**: 8가지 규칙으로 최종 판정 (STRONG_BUY ~ AVOID)

---

## 파이프라인 흐름도

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1: INGEST (데이터 수집)                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 1-1     │→│ 1-1b    │→│ 1-2     │→│ 1-3     │→│ 1-4     │        │
│  │ 테마    │  │ 종목    │  │ 주가    │  │ 재무    │  │ 뉴스    │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 2: PROCESSING (데이터 처리)                                        │
│  ┌─────────────────┐     ┌─────────────────────────────────────┐        │
│  │ 2-1 전처리       │ →  │ 2-2 섹터 라벨링 (LLM)                 │        │
│  │ HTML 제거/정제   │     │ 메인섹터 + 보조섹터 + TYPE_A/B 결정   │        │
│  └─────────────────┘     └─────────────────────────────────────┘        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 3: ANALYSIS (섹터 분석)                                            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐     ┌───────────┐         │
│  │ S_Flow    │ +│ S_Breadth │ +│ S_Trend   │  →  │ Tier 분류  │         │
│  │ 수급 강도  │  │ 결속력    │  │ 추세 점수  │     │ 1/2/3/Skip│         │
│  └───────────┘  └───────────┘  └───────────┘     └───────────┘         │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 4: FILTERING (필터링)                                              │
│  ┌─────────────────────────┐     ┌─────────────────────────┐           │
│  │ Track A (Hard Filter)   │ OR │ Track B (Soft Filter)    │           │
│  │ 실적형: PBR<3, 영업익>0  │     │ 성장형: 자본잠식<50%     │           │
│  └─────────────────────────┘     └─────────────────────────┘           │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 5: SCORING (점수화)                                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Track A: 재무(50%) + 기술(50%)                            │           │
│  │ Track B: 재무(20%) + 기술(80%)                            │           │
│  └──────────────────────────────────────────────────────────┘           │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 6: DECISION (최종 판정)                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐       │
│  │ Skeptic     │  +  │ Sentiment   │  →  │ Decision Matrix      │       │
│  │ 재료 분석    │     │ 심리 분석    │     │ STRONG_BUY/BUY/...   │       │
│  └─────────────┘     └─────────────┘     └─────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 각 단계별 상세

### Layer 1: 데이터 수집 (Ingest)

#### 1-1. 테마 목록 크롤링

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/naver_theme.py` |
| **데이터 소스** | 네이버 금융 테마 페이지 |
| **Input** | 없음 (페이지 수 지정 가능) |
| **Output** | `List[Theme]` - theme_id, name, change_rate |

#### 1-1b. 테마별 종목 수집

| 항목 | 내용 |
|------|------|
| **메서드** | `fetch_theme_stocks(theme_id, theme_name)` |
| **Input** | `theme_id` |
| **Output** | `List[ThemeStock]` - stock_code, stock_name, theme_name |

#### 1-2. 주가 데이터 수집

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/price_fetcher.py` |
| **데이터 소스** | pykrx (KRX 공식 데이터) |
| **Input** | `stock_codes`, `lookback_days=120` |
| **Output** | `Dict[code → DataFrame]` - OHLCV |

#### 1-3a. 재무 데이터 수집 (DART)

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/dart_client.py` |
| **데이터 소스** | DART OpenAPI |
| **Input** | `stock_code`, `year` |
| **Output** | `FinancialData` - 매출, 영업이익, 부채비율, R&D비율 등 |

#### 1-3b. 사업보고서 텍스트 수집 (DART)

| 항목 | 내용 |
|------|------|
| **메서드** | `fetch_business_report(stock_code, year)` |
| **용도** | 섹터 라벨링 시 보조 섹터 결정 |
| **Output** | `str` - 사업의 내용 텍스트 |

#### 1-3c. 수급/펀더멘탈 수집 (pykrx)

| 항목 | 내용 |
|------|------|
| **메서드** | `fetch_stock_supply_demand()`, `fetch_stock_fundamental()` |
| **Output** | 외인/기관 순매수, 시총, PBR, PER |

#### 1-4. 뉴스 크롤링

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/news_crawler.py` |
| **데이터 소스** | 네이버 금융 뉴스 탭 |
| **Output** | `List[NewsArticle]` - title, source, published_at |

#### 1-5. 토론방 크롤링 (NEW - Sentiment B)

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/discussion_crawler.py` |
| **데이터 소스** | 네이버 금융 토론방 |
| **Output** | `Dict[code → {community_posts, sentiment_ratio, likes, dislikes}]` |
| **용도** | Sentiment Reader B 접근법 (투자자 심리 분석) |

---

### Layer 2: 데이터 처리 (Processing)

#### 2-1. 전처리

| 항목 | 내용 |
|------|------|
| **파일** | `src/processing/preprocessor.py` |
| **처리** | HTML 태그 제거, 특수문자 정제, 중복 제거 |

#### 2-2. 섹터 라벨링 (핵심)

| 항목 | 내용 |
|------|------|
| **파일** | `src/processing/sector_labeler.py` |
| **Input** | theme_names, dart_business_text, news_headlines |
| **Output** | `SectorLabel` |

**섹터 라벨링 로직 (수정됨):**
```
1. primary_sector: 네이버 테마에서 직접 결정 (투자자 관심 기준)
   - 테마 키워드 매칭으로 섹터 분류
   - 종목명에서 힌트 추출 (보조)

2. secondary_sectors: DART + 뉴스를 LLM으로 분석하여 결정 (실제 사업 기준)
   - DART 사업보고서에서 사업 키워드 추출
   - 뉴스 헤드라인에서 현재 이슈 추출
   - LLM이 primary 제외한 관련 섹터 2개까지 추출

3. is_growth_sector: 성장 섹터 여부 → TYPE_A/B 결정
```

**Output 구조:**
```python
SectorLabel:
  - primary_sector: str       # 주요 섹터 (네이버 테마 기반)
  - secondary_sectors: list   # 보조 섹터 (DART/뉴스 LLM 분석)
  - theme_tags: list          # 원본 테마 태그
  - business_keywords: list   # DART에서 추출한 키워드
  - current_issues: list      # 뉴스에서 추출한 현재 이슈
  - is_growth_sector: bool    # TYPE_B 여부
  - confidence: float
```

**성장 섹터 (TYPE_B) 목록:**
- 반도체, 2차전지, 바이오/제약, IT/소프트웨어
- 헬스케어, 방산, 에너지

---

### Layer 3: 섹터 분석 (Analysis)

#### 3개 핵심 지표

| 지표 | 공식 | 의미 |
|------|------|------|
| **S_Flow** | (외인순매수 + 기관순매수×1.2) / 시총 × 100 | 수급 강도 |
| **S_Breadth** | MA20 이상 종목 수 / 전체 × 100 | 결속력 |
| **S_Trend** | 정배열(50) + 모멘텀(30) - 과열(20) | 추세 점수 |

#### Tier 분류 매트릭스

```
                S_Trend (추세)
             Low/Flat       High
         ┌─────────────┬─────────────┐
S_Flow   │   TIER 1    │   TIER 2    │  ← 투자 가능
 High    │  수급 빈집   │  주도 섹터   │
         ├─────────────┼─────────────┤
S_Flow   │    SKIP     │   TIER 3    │  ← 회피
 Low     │   무관심     │  가짜 상승   │
         └─────────────┴─────────────┘
```

| Tier | 조건 | 의미 | 액션 |
|------|------|------|------|
| **TIER 1** | 수급↑ + 추세↓ | 돈은 들어오는데 아직 안 올랐음 | 선취매 (최고 기회) |
| **TIER 2** | 수급↑ + 추세↑ | 돈도 들어오고 차트도 좋음 | 눌림목 대기 |
| **TIER 3** | 수급↓ + 추세↑ | 차트만 올랐고 돈은 안 들어옴 | 진입 금지 (가짜) |
| **SKIP** | 수급↓ + 추세↓ | 관심 없는 섹터 | 관망 |

**대장주 착시 검증:** 대장주 1개 제외 후에도 S_Flow > 0 이어야 Tier 1,2 유지

---

### Layer 4: 필터링 (Filtering)

#### Track A (Hard Filter) - 실적형 섹터

| 조건 | 기준 | 탈락 사유 |
|------|------|----------|
| 영업이익 | 4분기 합산 > 0 | 영업이익 적자 |
| 부채비율 | < 200% | 부채비율 초과 |
| PBR | < 3.0 | PBR 과열 |
| 거래대금 | > 10억 | 거래대금 부족 |

#### Track B (Soft Filter) - 성장형 섹터

| 조건 | 기준 | 비고 |
|------|------|------|
| 자본잠식률 | < 50% | 필수 |
| 유동비율 | > 100% | 필수 |
| R&D 비중 | > 5% | 가산점만 |
| 거래대금 | > 5억 | 기준 완화 |

---

### Layer 5: 점수화 (Scoring)

#### Track별 가중치

| Track | 재무 | 기술 | 대상 |
|-------|------|------|------|
| Track A | 50% | 50% | 실적형 섹터 |
| Track B | 20% | 80% | 성장형 섹터 |

#### 재무 점수 구성

**Track A:**
- 영업이익 YoY (30%): -20% ~ +50% 범위 정규화
- ROE (30%): 0% ~ 20% 범위 정규화
- 부채비율 패널티 (10%): 감점 방식

**Track B:**
- 매출액 YoY (20%): -10% ~ +100% 범위 정규화
- R&D 비중 (10%): 0% ~ 20% 범위 정규화

#### 기술 점수 구성 (공통)

| 지표 | 가중치 | 설명 |
|------|--------|------|
| 수급 | 35% | 외인 + 기관×1.2 정규화 |
| MA20 이격도 | 20% | 눌림목(-10~0%)이 최고점 |
| 거래량 증가율 | 20% | 0~200% 범위 정규화 |
| 52주 신고가 근접도 | 25% | 50~100% 범위 정규화 |

---

### Layer 6: 최종 판정 (Decision)

#### Skeptic 분석 (재료 분석)

| 항목 | 내용 |
|------|------|
| **파일** | `src/decision/personas/skeptic.py` |
| **Input** | 뉴스 헤드라인, DART 공시 |
| **Output** | `MaterialGrade (S/A/B/C)` |

**재료 등급:**
- **S급**: 정부 정책, 대규모 수주, FDA 승인, 대형 M&A
- **A급**: 실적 개선, 신규 계약, 기술 개발
- **B급**: 일반 호재, 업황 개선 기대
- **C급**: 재료 없음, 악재 존재

#### Sentiment Reader 분석 (A+B+C 통합 심리 분석)

| 항목 | 내용 |
|------|------|
| **파일** | `src/decision/personas/sentiment.py` |
| **방법론** | A+B+C 세 가지 소스 통합 분석 |
| **Output** | `SentimentStage` + 개별 분석 결과 |

**A+B+C 통합 접근법:**

| 접근법 | 데이터 소스 | 분석 내용 | 가중치 |
|--------|------------|----------|--------|
| **A (뉴스 기반)** | 뉴스 헤드라인 | 뉴스 수량, 헤드라인 어조 분석 | 30% |
| **B (토론방 기반)** | 네이버 금융 토론방 | 공감/비공감 비율, 글 내용 분석 | 40% |
| **C (가격 기반)** | 주가/거래량 데이터 | RSI, 수익률, 거래량 패턴 | 30% |

**심리 단계:**
- **공포(FEAR)**: 대중 무관심, 바닥권 (매수 기회)
- **의심(DOUBT)**: 초기 관심, 반신반의
- **확신(CONVICTION)**: 상승 확신 중기
- **환희(EUPHORIA)**: 과열 (매도 시점)

**Output 구조:**
```python
SentimentAnalysis:
  - sentiment_stage: SentimentStage  # 종합 판정
  - news_stage: SentimentStage       # A: 뉴스 기반
  - discussion_stage: SentimentStage # B: 토론방 기반
  - price_stage: SentimentStage      # C: 가격 기반
  - confidence: float
  - interest_level: float
  - tone_score: float
  - rsi: float
  - volume_ratio: float
```

#### Decision Matrix

| 조건 | 결과 |
|------|------|
| 환희 상태 | → **AVOID** |
| 재료 C급 | → **WATCH** |
| Tier1 + S/A급 + 공포/의심 | → **STRONG_BUY** |
| Tier1 + S/A급 + 확신 | → **BUY** |
| Tier2 + A/B급 + 의심/확신초기 | → **BUY** |
| Tier3 | → **AVOID** |
| 그 외 | → **WATCH** |

---

## 핵심 개념

### 섹터 타입 (TYPE_A vs TYPE_B)

| 구분 | TYPE_A (실적형) | TYPE_B (성장형) |
|------|----------------|----------------|
| 대상 | 금융, 화학, 철강, 건설 등 | 반도체, 바이오, 2차전지 등 |
| 필터 | Hard Filter (엄격) | Soft Filter (유연) |
| 가중치 | 재무 50% : 기술 50% | 재무 20% : 기술 80% |

### Tier 시스템

- **Tier 1**: 수급 빈집 → 선취매 기회 (최고)
- **Tier 2**: 주도 섹터 → 눌림목 대기
- **Tier 3**: 가짜 상승 → 진입 금지

### 대장주 착시 검증

대장주 1개가 섹터 전체 수급을 왜곡하는 현상을 방지합니다.
대장주 제외 후에도 S_Flow > 0 이어야 정상으로 판정합니다.

---

## 프로젝트 구조

```
src/
├── core/                    # 핵심 인프라
│   ├── config.py           # 설정 관리
│   ├── database.py         # DB 연결
│   ├── models.py           # ORM 모델
│   └── interfaces.py       # 인터페이스 정의
│
├── ingest/                  # Layer 1: 데이터 수집
│   ├── naver_theme.py      # 테마 크롤링
│   ├── price_fetcher.py    # 주가/수급 (pykrx)
│   ├── dart_client.py      # DART API
│   ├── news_crawler.py     # 뉴스 크롤링
│   └── discussion_crawler.py # 토론방 크롤링 (Sentiment B)
│
├── processing/              # Layer 2: 데이터 처리
│   ├── preprocessor.py     # 텍스트 전처리
│   └── sector_labeler.py   # 섹터 라벨링 (LLM)
│
├── analysis/                # Layer 3: 섹터 분석
│   ├── tier_classifier.py  # Tier 분류
│   └── metrics/
│       ├── flow.py         # S_Flow 계산
│       ├── breadth.py      # S_Breadth 계산
│       └── trend.py        # S_Trend 계산
│
├── filtering/               # Layer 4: 필터링
│   ├── filter_router.py    # 필터 라우팅
│   └── track_filters.py    # Track A/B 필터
│
├── scoring/                 # Layer 5: 점수화
│   └── stock_scorer.py     # 종목 점수 산출
│
├── decision/                # Layer 6: 최종 판정
│   ├── decision_engine.py  # Decision Matrix
│   └── personas/
│       ├── skeptic.py      # 재료 분석
│       └── sentiment.py    # 심리 분석
│
├── llm/                     # LLM 클라이언트
│   └── ollama_client.py    # Ollama 로컬 LLM
│
└── orchestrator/            # 오케스트레이션
    ├── pipeline.py         # 전체 파이프라인
    └── stage_runner.py     # 단계별 실행기
```

---

## 설치 및 실행

### 요구사항

- Python 3.11+
- DART API 키 (환경변수: `DART_API_KEY`)
- Ollama (로컬 LLM 사용 시)

### 설치

```bash
pip install -r requirements.txt
```

### 실행

```python
from src.orchestrator import Pipeline

pipeline = Pipeline()
result = pipeline.run_full(
    theme_codes=["AI반도체", "2차전지"],
    year=2025,
    max_stocks=100,
    save_to_db=True
)

print(f"추천 종목: {len(result.final_recommendations)}개")
```

### 테스트

```bash
pytest tests/
```

---

## 라이선스

MIT License
