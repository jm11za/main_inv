# 프로젝트 진행 상황

> 마지막 업데이트: 2026-02-01 (v3.0)

---

## 📊 전체 진행률

```
Core 모듈      [██████████] 100%
Layer 1        [██████████] 100%
Layer 2        [██████████] 100%
Layer 3        [██████████] 100%
Layer 3.5      [██████████] 100%
Layer 4        [██████████] 100%
Layer 5        [██████████] 100%
Layer 6        [██████████] 100%
Orchestrator   [██████████] 100%
```

---

## ✅ 완료된 작업

### Core 모듈
- [x] `config.py` - YAML 설정 로드, 환경별 오버라이드
- [x] `logger.py` - loguru 기반 구조화 로깅
- [x] `database.py` - SQLAlchemy 커넥션 풀, 세션 관리
- [x] `cache.py` - 인메모리 캐시, TTL 지원
- [x] `exceptions.py` - 레이어별 커스텀 예외
- [x] `interfaces.py` - Enum, DataClass, Abstract 인터페이스
- [x] `models.py` - ORM 모델 (Theme, Stock, DailyPrice, CrawlHistory)

### Layer 1: Data Ingest
- [x] `PriceDataFetcher` - pykrx 기반 OHLCV, 수급, 시가총액
- [x] `NaverThemeCrawler` - 테마 목록 + 소속 종목 크롤링 (개선됨)
  - Anti-blocking: User-Agent 회전, 불규칙 딜레이, 지수 백오프
  - 전체 페이지 수집 지원, 403/429 에러 자동 처리
- [x] `ThemeService` - 테마 데이터 DB 저장/조회
- [x] `DartApiClient` - 재무제표 + 사업보고서 조회 (확장됨)
  - 4분기 합산 영업이익, 자본잠식률, R&D 비중 계산
- [x] `NewsCrawler` - 종목별 뉴스 헤드라인 수집
- [x] `CommunityCrawler` - 네이버 종목토론실 크롤링 (Sentiment용)
  - Anti-blocking: User-Agent 회전, 지수 백오프, 자동 쿨다운

### LLM 모듈
- [x] `ClaudeClient` - Anthropic API 클라이언트 구조 (인증 이슈로 미사용)
- [x] `ClaudeCliClient` - subprocess 기반 Claude CLI 호출 (현재 사용)
- [x] `OllamaClient` - 로컬 LLM (deepseek-r1:14b, llama3.1:8b 등)

### Layer 2: Processing (데이터 처리) ✅
- [x] `Preprocessor` - HTML 제거, 정규화, 중복 제거
- [x] `LLMExtractor` - 뉴스/DART에서 키워드 추출 (Ollama + 폴백)
- [x] `SynonymResolver` - 동의어 표준화 (이차전지→2차전지 등)
- [x] `TagMapper` - Theme-Stock 매핑 통합
- [x] `DataTransformer` - Layer 간 데이터 변환
  - 필터용 재무 데이터 통합 조회
  - 수급 데이터 → S_Flow 입력 변환
  - FilterResult → StockScorer 입력 변환
- [x] `StockThemeAnalyzer` - 섹터 분류 + 부가정보 (v3.0)
  - 섹터: 네이버 테마명 그대로 사용 (N:M 관계)
  - 부가정보: DART 사업개요 + 뉴스 LLM 요약
  - "메인/보조 섹터" 개념 폐기 → 단순히 "섹터" (N개 가능)
- [x] 테스트 완료: 59/59 PASSED

### Layer 3: Analysis (분석) ✅
- [x] `FlowCalculator` - S_Flow 수급 강도 (외인+기관 순매수/시총)
- [x] `BreadthCalculator` - S_Breadth 내부 결속력 (MA20 위 종목 비율)
- [x] `TrendCalculator` - S_Trend 추세 점수 (정배열+모멘텀-과열)
- [x] `TierClassifier` - 섹터 Tier 분류 (Tier 1/2/3/SKIP)
- [x] 대장주 착시 감지 로직
- [x] 테스트 완료: 24/24 PASSED

### Layer 3.5: Filtering (이원화 필터링) ✅
- [x] `SectorClassifier` - 규칙 기반 + LLM 폴백 섹터 분류
- [x] `TrackAFilter` - Hard Filter (영업이익, 부채비율, PBR, 거래대금)
- [x] `TrackBFilter` - Soft Filter (자본잠식, 유동비율, R&D 가산점)
- [x] `FilterRouter` - 섹터별 필터 라우팅
- [x] 테스트 완료: 25/25 PASSED

### Layer 4: Scoring (종목 점수화) ✅
- [x] `StockScorer` - Track별 가중치 적용 점수 산출
- [x] `FinancialMetrics` / `TechnicalMetrics` - 지표 데이터클래스
- [x] `ScoreResult` - 점수 결과 (financial, technical, total, breakdown)
- [x] Track A: 재무 50% + 기술적 50%
- [x] Track B: 재무 20% + 기술적 80%
- [x] 테스트 완료: 19/19 PASSED

### Layer 5: Decision (최종 판정) ✅
- [x] `Skeptic` - 냉철한 애널리스트 (재료 분석 → S/A/B/C 등급)
- [x] `SentimentReader` - 심리 분석가 (대중 심리 → 공포/의심/확신/환희)
- [x] `DecisionEngine` - Decision Matrix 기반 최종 판정
- [x] `LLMAnalyzer` - Dual Persona 조율자
- [x] 환희 → AVOID, 재료 C급 → WATCH 로직
- [x] 테스트 완료: 26/26 PASSED

---

## 🔄 진행 중인 작업

### 파이프라인 통합 ✅
- [x] 수급 DataFrame → 개별 값 변환 로직 (DataTransformer)
- [x] 영업이익 4Q 합산 로직 (DartApiClient)
- [x] 자본잠식률 계산 로직 (DartApiClient)
- [x] R&D 비중 추출 로직 (DartApiClient)
- [x] FilterResult → StockScorer 입력 변환 (DataTransformer)
- [x] Orchestrator 모듈 구현 ✅
- [x] Layer 6 (Output) 구현 ✅

### Orchestrator 모듈 ✅
- [x] `StageRunner` - 개별 단계 실행기
  - `run_sector_labeling()` 추가 (종합 섹터 라벨링)
- [x] `Pipeline` - 전체 파이프라인 조율
  - SectorLabeling 단계 통합
- [x] `run_full()` - 전체 분석 실행
- [x] `run_quick()` - 빠른 분석 (테마 수집 생략)
- [x] 테스트 완료: 14/14 PASSED

### Layer 6: Output ✅
- [x] `ReportGenerator` - Claude 기반 리포트 생성
- [x] `AnalysisReport` - 분석 리포트 데이터클래스
- [x] `TelegramNotifier` - Telegram 알림 발송
- [x] 텍스트/Telegram 포맷 지원
- [x] 테스트 완료: 19/19 PASSED

---

## 📋 다음 단계

1. ~~**Telegram Chat ID 설정**~~ - 완료 (Chat ID: 1124186331)
2. **End-to-End 테스트** - 전체 파이프라인 실행 테스트
3. **파이프라인 최적화** - 순차적 개선 완료
   - [x] 1-1: NaverThemeCrawler 전체 페이지 + Anti-blocking
   - [x] 1-2: SectorLabeler 종합 라벨링 (메인/보조 섹터)
   - [x] 2-1: SectorLabeler → Pipeline 통합 + DB 저장
   - [x] 2-2: PBR, 거래대금 pykrx 조회 추가
   - [x] 2-3: 개별 종목 S_Flow 계산 추가
   - [x] 2-4: DART 사업보고서 텍스트 추출 개선
   - [ ] 3-1: 전체 파이프라인 통합 테스트

---

## 🔧 주요 결정 사항

| 날짜 | 결정 내용 |
|------|----------|
| 2026-01-31 | 테마 데이터: 초기 전체 수집 후 주 1회 업데이트 방식 |
| 2026-01-31 | 주가 데이터: pykrx 사용 (무료, API 키 불필요) |
| 2026-01-31 | DART API: 재무제표 + 사업보고서 모두 수집 |
| 2026-01-31 | 섹터 분류: 규칙 기반 우선, LLM 폴백 (개별 분류) |
| 2026-01-31 | Claude CLI: subprocess 방식으로 현재 OAuth 토큰 활용 |
| 2026-01-31 | 세션 지속성: PROJECT_STATUS.md로 컨텍스트 유지 |
| 2026-01-31 | 키워드 추출: Ollama (deepseek-r1:14b) + 규칙 폴백 |
| 2026-01-31 | 추출 대상: 뉴스 + DART 사업보고서 모두 |
| 2026-01-31 | 자본잠식률: (자본금 - 총자본) / 자본금 * 100 |
| 2026-01-31 | R&D 비중: 연구개발비 / 매출액 * 100 |
| 2026-01-31 | LLM 타임아웃: 300초로 증가 (deepseek-r1:14b 추론 시간 고려) |
| 2026-01-31 | SectorLabeler: 테마(1.5x) + DART(2.0x) + 뉴스(1.0x) 가중치 |
| 2026-01-31 | NaverThemeCrawler: 불규칙 딜레이 1.5~4초, 재시도 3회 |
| 2026-02-01 | Stage 1: 섹터(N:M) + 부가정보(DART/뉴스 LLM 요약) |
| 2026-01-31 | 개별 종목 S_Flow: FlowCalculator.calculate_stock_auto() |
| 2026-01-31 | 섹터 라벨 DB 저장: StockModel에 primary_sector, secondary_sectors 필드 |
| 2026-01-31 | PBR/거래대금: pykrx에서 실시간 조회 (하드코딩 제거) |

---

## 🔑 환경 설정

### .env 파일 필요 항목
```
DART_API_KEY=발급완료
ANTHROPIC_API_KEY=불필요 (Claude CLI subprocess 사용)
```

### 설치된 주요 패키지
- pandas, numpy, requests, beautifulsoup4
- pykrx (KRX 주가 데이터)
- sqlalchemy, pyyaml, loguru
- pytest, pylint, anthropic

---

## 🐛 알려진 이슈

1. **Claude API 인증**: OAuth 토큰은 Python SDK에서 직접 사용 불가. → `ClaudeCliClient` (subprocess)로 해결됨.
2. **Ollama 서버**: 사용 전 `ollama serve` 실행 필요. 미실행 시 규칙 기반 폴백 사용.
3. **Ollama localhost 연결**: `localhost`가 IPv6로 해석되어 연결 실패 → `127.0.0.1`로 수정하여 해결됨.

---

## 📁 프로젝트 구조

```
main_inv/
├── config/           # 설정 파일
├── src/
│   ├── core/         # 공통 모듈 ✅
│   ├── ingest/       # Layer 1 ✅
│   ├── processing/   # Layer 2 ✅ (DataTransformer 추가)
│   ├── analysis/     # Layer 3 ✅
│   │   └── metrics/  # S_Flow, S_Breadth, S_Trend
│   ├── filtering/    # Layer 3.5 ✅
│   ├── scoring/      # Layer 4 ✅
│   ├── decision/     # Layer 5 ✅
│   ├── orchestrator/ # 파이프라인 조율 ✅
│   ├── output/       # Layer 6 ✅
│   └── llm/          # LLM Gateway ✅
├── tests/
├── .env              # API 키 (gitignore)
└── requirements.txt
```
