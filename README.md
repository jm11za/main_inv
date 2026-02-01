# 주식 분석 파이프라인 (Stock Analysis Pipeline)

테마 기반 종목 발굴 및 투자 판정 시스템

> **v3.0 (2026-02-01)**: SectorCategory Enum 제거 → 네이버 테마명 문자열 직접 사용, N:M 종목-테마 관계 지원

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

## 파이프라인 흐름도 (v3.0)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 0: 데이터 수집 (Data Collect)                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ 0-1     │  │ 0-2     │  │ 0-3     │  │ 0-4     │  │ 0-5     │        │
│  │ 테마    │  │ 주가    │  │ 재무    │  │ 수급    │  │ 뉴스    │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1: 섹터 분류 (StockThemeAnalyzer)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  테마-종목 매핑 구축 (N:M 관계)                                     │    │
│  │  • theme_stocks_map: {테마명 → [종목코드, ...]}                     │    │
│  │  • stock_themes_map: {종목코드 → [테마명, ...]}                     │    │
│  │  • LLM 사업/뉴스 요약 생성                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 2: 섹터 타입 분류 (SectorTypeAnalyzer)                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  키워드 기반 + LLM 분류 → Type A(실적형) / Type B(성장형)           │    │
│  │  Output: {테마명: SectorType} (예: {"반도체": TYPE_B})              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 3: 섹터 우선순위 (SectorPrioritizer)                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐     ┌───────────┐         │
│  │ S_Flow    │ +│ S_Breadth │ +│ S_Trend   │  →  │ 테마 순위  │         │
│  │ 수급 강도  │  │ 결속력    │  │ 추세 점수  │     │ + LLM 전망 │         │
│  └───────────┘  └───────────┘  └───────────┘     └───────────┘         │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 4: 종목 선정 (CandidateSelector)                                   │
│  ┌─────────────────────────┐     ┌─────────────────────────┐           │
│  │ Track A (Hard Filter)   │ OR │ Track B (Soft Filter)    │           │
│  │ + 점수화 (재무50:기술50) │     │ + 점수화 (재무20:기술80) │           │
│  └─────────────────────────┘     └─────────────────────────┘           │
│  → 테마당 상위 N개, 전체 최대 M개 선정                                    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 5: 최종 검증 (MaterialAnalyzer + SentimentAnalyzer)                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐       │
│  │ Skeptic     │  +  │ Sentiment   │  →  │ DecisionEngine       │       │
│  │ 재료 분석    │     │ 심리 분석    │     │ STRONG_BUY/BUY/...   │       │
│  │ (Claude)    │     │ (Claude)    │     │                     │       │
│  └─────────────┘     └─────────────┘     └─────────────────────┘       │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 6: 아웃풋 (ReportGenerator + TelegramNotifier)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ StageSaver      │  │ ReportGenerator │  │ TelegramNotifier│         │
│  │ 단계별 JSON 저장 │  │ 리포트 생성     │  │ 텔레그램 발송    │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 각 단계별 상세 (v3.0)

### Stage 0: 데이터 수집 (Data Collect)

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

#### 1-4. 뉴스 크롤링 (v3.1 개선)

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/naver_news_search.py` |
| **클래스** | `NaverNewsSearchCrawler` |
| **데이터 소스** | 네이버 통합검색 뉴스 탭 (종목명 기반 검색) |
| **기간 필터** | 1일, 1주, 1개월, 6개월, 1년 |
| **Output** | `List[NaverNewsArticle]` - title, summary, press, published_at, link |
| **딜레이** | 4초 (API 차단 방지) |
| **LLM 프롬프트** | 관련성 낮은 기사 자동 필터링 지시 포함 |

**주요 특징:**
- 종목명 기반 검색 (종목코드 아님)
- 헤드라인 + 요약(미리보기) 동시 수집
- 언론사, 발행일 자동 추출
- 기간별 필터링 지원

#### 1-5. 토론방 크롤링 (NEW - Sentiment B)

| 항목 | 내용 |
|------|------|
| **파일** | `src/ingest/discussion_crawler.py` |
| **데이터 소스** | 네이버 금융 토론방 |
| **Output** | `Dict[code → {community_posts, sentiment_ratio, likes, dislikes}]` |
| **용도** | Sentiment Reader B 접근법 (투자자 심리 분석) |

---

### Stage 1: 섹터 분류 (StockThemeAnalyzer)

| 항목 | 내용 |
|------|------|
| **파일** | `src/sector/classifier.py` |
| **클래스** | `StockThemeAnalyzer` |

**역할:** 종목별 섹터(테마) 할당 + 부가정보(DART/뉴스 LLM 요약) 생성

**INPUT:**
```
- themes: 네이버 테마 목록 (ThemeInfo[])
- theme_stocks: 테마-종목 매핑 (ThemeStock[])
- stock_codes: 분석 대상 종목코드
- stock_names: {종목코드: 종목명}
- dart_data: {종목코드: DART 사업개요 텍스트}
- news_data: {종목코드: [뉴스 기사]}
```

**OUTPUT:**
```
- theme_stocks_map: {테마명: [종목코드,...]}
- stock_themes_map: {종목코드: [테마명,...]}  ← N:M 관계
- stocks_data: StockAnalysisData[]
```

**StockAnalysisData 구조:**
```python
@dataclass
class StockAnalysisData:
    stock_code: str
    stock_name: str

    # 섹터 (N개 가능 - 1종목이 여러 테마에 속할 수 있음)
    theme_tags: list[str]     # ["반도체", "AI", "HBM"]

    # 부가정보 (LLM 요약)
    business_summary: str     # DART 사업개요 요약 (3-5문장)
    news_summary: str         # 뉴스 동향 요약 (3-5문장)

    # 메타데이터
    data_sources: list[str]   # ["theme", "dart", "news"]
```

**핵심 원칙:**
- 섹터 = 네이버 테마명 그대로 사용 (Enum 없음)
- 1종목 N섹터 관계 (N:M) 지원
- 부가정보 = DART 사업개요 + 뉴스를 LLM으로 요약

---

### Stage 2: 섹터 타입 분류 (SectorTypeAnalyzer)

| 항목 | 내용 |
|------|------|
| **파일** | `src/sector/type_analyzer.py` |
| **클래스** | `SectorTypeAnalyzer` |
| **Input** | 테마명 문자열 리스트 (예: ["반도체", "은행", "바이오"]) |
| **Output** | `{테마명: SectorType}` 매핑 |

**분류 로직:**
1. 키워드 매칭 (우선)
   - TYPE_B_KEYWORDS: 반도체, HBM, AI, 2차전지, 바이오, 신약, 로봇 등
   - TYPE_A_KEYWORDS: 은행, 건설, 철강, 자동차, 음식료 등
2. LLM 분류 (키워드 미매칭 시)

---

### Stage 3: 섹터 우선순위 (SectorPrioritizer)

| 항목 | 내용 |
|------|------|
| **파일** | `src/sector/prioritizer.py` |
| **클래스** | `SectorPrioritizer`, `ThemeMetrics` |
| **Input** | `ThemeMetrics[]` (테마별 지표) |
| **Output** | 상위 N개 테마 선정 (점수 + LLM 전망 분석) |

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

### Stage 4: 종목 선정 (CandidateSelector)

| 항목 | 내용 |
|------|------|
| **파일** | `src/stock/selector.py` |
| **클래스** | `CandidateSelector`, `CandidateResult` |
| **Input** | `stocks_data[]`, `sector_type_map: {테마명: SectorType}` |
| **Output** | 테마별 상위 N개, 전체 최대 M개 선정 |

**프로세스:**
1. 테마별 그룹핑 (테마명 문자열 기준)
2. Track A/B 필터 적용
3. 점수화 (재무 + 기술)
4. 테마 내 순위 → 전체 순위

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

#### 점수화 (StockScorer)

| 항목 | 내용 |
|------|------|
| **파일** | `src/stock/scorer.py` |
| **클래스** | `StockScorer`, `StockScoreResult` |

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

### Stage 5: 최종 검증 (MaterialAnalyzer + SentimentAnalyzer + DecisionEngine)

| 항목 | 내용 |
|------|------|
| **파일** | `src/verify/material_analyzer.py`, `src/verify/sentiment_analyzer.py`, `src/verify/decision_engine.py` |
| **LLM** | Claude (Cloud) - 정밀 분석 필요 |
| **Input** | 선정된 종목 + 뉴스 + 커뮤니티 데이터 |
| **Output** | `STRONG_BUY / BUY / WATCH / AVOID` |

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

### Stage 6: 아웃풋 (ReportGenerator + TelegramNotifier)

| 항목 | 내용 |
|------|------|
| **파일** | `src/output/stage_saver.py`, `src/output/report_generator.py`, `src/output/telegram_notifier.py` |

#### 출력물

| 출력물 | 파일 경로 | 내용 |
|--------|----------|------|
| 단계별 결과 | `outputs/stages/{날짜}_0X_*.json` | 각 스테이지 결과 JSON |
| 일일 리포트 | `outputs/reports/daily/{날짜}_full.json` | 전체 집계 결과 |
| 텔레그램 메시지 | `outputs/reports/telegram/{날짜}_summary.txt` | 발송용 메시지 |

#### 텔레그램 발송

```python
from src.output import TelegramNotifier, ReportGenerator

# 리포트 생성
generator = ReportGenerator()
report = generator.generate_from_stages(aggregated_data)

# 발송
notifier = TelegramNotifier()
notifier.send_report(report)
```

---

## 핵심 개념

### 섹터 타입 (TYPE_A vs TYPE_B) - v3.0

| 구분 | TYPE_A (실적형) | TYPE_B (성장형) |
|------|----------------|----------------|
| 대상 | 금융, 화학, 철강, 건설, 자동차 등 | 반도체, 바이오, 2차전지, AI, 로봇 등 |
| 필터 | Hard Filter (엄격) | Soft Filter (유연) |
| 가중치 | 재무 50% : 기술 50% | 재무 20% : 기술 80% |
| 분류 방식 | 테마명 키워드 매칭 + LLM | 테마명 키워드 매칭 + LLM |

**v3.0 변경**: SectorCategory Enum 대신 테마명 문자열로 분류
```python
# 예시: sector_type_map
{
    "반도체": SectorType.TYPE_B,
    "2차전지": SectorType.TYPE_B,
    "은행": SectorType.TYPE_A,
    "자동차": SectorType.TYPE_A,
}
```

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
│   ├── news_crawler.py     # 뉴스 크롤링 (레거시)
│   ├── naver_news_search.py # 뉴스 검색 크롤러 (v3.1 - 종목명 기반, 요약 포함)
│   └── discussion_crawler.py # 토론방 크롤링 (Sentiment B)
│
├── processing/              # Layer 2: 데이터 처리
│   ├── preprocessor.py     # 텍스트 전처리
│   ├── llm_extractor.py    # LLM 키워드 추출
│   ├── tag_mapper.py       # 테마-종목 매핑
│   └── data_transformer.py # 데이터 변환
│
├── sector/                  # Layer 3: 섹터/테마 분석 (v3.0)
│   ├── classifier.py       # ① StockThemeAnalyzer - 테마 데이터셋 구축
│   ├── type_analyzer.py    # ② SectorTypeAnalyzer - Type A/B 분류
│   └── prioritizer.py      # ③ SectorPrioritizer - 테마 우선순위
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
├── stock/                   # Layer 4: 종목 분석 (v3.0)
│   ├── filter.py           # StockFilter - Track A/B 필터
│   ├── scorer.py           # StockScorer - 종목 점수화
│   └── selector.py         # CandidateSelector - 후보 선정
│
├── llm/                     # LLM 클라이언트
│   ├── ollama_client.py    # Ollama 로컬 LLM
│   └── claude_client.py    # Claude API 클라이언트
│
├── output/                  # Layer 6: 결과 출력
│   ├── stage_saver.py      # 단계별 결과 저장
│   ├── report_generator.py # 리포트 생성
│   └── telegram_notifier.py # 텔레그램 알림
│
└── orchestrator/            # 오케스트레이션
    ├── pipeline.py         # 전체 파이프라인
    ├── pipeline_v2.py      # v3.0 파이프라인
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
# v3.0 파이프라인 사용
from src.orchestrator.pipeline_v2 import PipelineV2

pipeline = PipelineV2()
result = pipeline.run_full(verbose=True)

# 결과 확인
print(f"분석 테마: {len(result.stage_results.get('sector_classify', {}).get('themes', []))}개")
print(f"선정 종목: {len(result.final_recommendations)}개")

# 텔레그램 발송
from src.output import TelegramNotifier, ReportGenerator

generator = ReportGenerator()
report = generator.generate_from_stages(result.to_dict())

notifier = TelegramNotifier()
notifier.send_report(report)
```

### 테스트

```bash
pytest tests/
```

---

## v3.0 변경사항 요약

### 핵심 변경
- **SectorCategory Enum 제거**: 네이버 테마명 문자열 직접 사용
- **N:M 관계 지원**: 한 종목이 여러 테마에 소속 가능
- **모듈 구조 개편**: `sector/` 패키지로 테마 분석 기능 통합

### 변경된 클래스
| AS-IS | TO-BE | 설명 |
|-------|-------|------|
| `SectorLabeler` | `StockThemeAnalyzer` | 테마 데이터셋 구축 |
| `SectorMetrics` | `ThemeMetrics` | 테마별 지표 (별칭 유지) |
| `sector: SectorCategory` | `sector: str` | 테마명 문자열 |

### 테스트 커버리지
- `tests/sector/test_classifier.py`: 31개
- `tests/sector/test_type_analyzer.py`: 27개
- `tests/sector/test_prioritizer.py`: 23개
- `tests/stock/test_selector.py`: 20개
- `tests/test_output.py`: 37개

---

## 라이선스

MIT License
