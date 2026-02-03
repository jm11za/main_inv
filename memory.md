# Session Memory

## 세션: 2026-02-02

### 완료된 작업
1. **Git 정리**
   - main 브랜치에 이 세션 커밋만 유지 (5개)
   - 불필요한 테스트 파일 삭제 (test_*.py, debug_*.py 등)
   - 중복 CLAUDE.md 삭제 (main_inv 폴더 내)

2. **인코딩 버그 수정**
   - Windows cp949 인코딩 문제 해결
   - 유니코드 문자 → ASCII 변환 (✓→[OK], ✗→[FAIL], █→#, ░→-)
   - pipeline_v2.py에 UTF-8 출력 설정 추가

3. **Pipeline Stage 0~2 테스트**
   - Stage 0: 데이터 수집 (테마, 종목, 주가, 재무, 뉴스, 사업개요)
   - Stage 1: 섹터 분류 (사업요약/뉴스요약 LLM 생성)
   - Stage 2: Type A/B 분류 (키워드 매칭 + LLM 배치 분석)
   - 테스트 결과: 모두 정상 작동

### 커밋 히스토리 (origin/main 대비 +5)
```
6161d45 chore: 중복 CLAUDE.md 파일 삭제
5604a19 feat: 파이프라인 필수 모듈 및 테스트 코드 추가
8df8c69 fix: Stage 2 키워드 매칭 및 출력 버그 수정
eeec955 fix: 테마 제한 시 종목 필터링 버그 수정 및 뉴스 크롤러 통합
20c6e71 feat: Pipeline v2.0 with theme-based analysis and debug mode
```

### 미커밋 변경사항
- `src/orchestrator/pipeline_v2.py` - UTF-8 출력 설정 추가
- `src/orchestrator/stage_runner.py` - ASCII 문자 변경 ([OK], #, -)
- `src/core/preflight.py` - ASCII 문자 변경 ([OK], [FAIL])

### 현재 환경
- **로컬 LLM**: llama3.1:8b-instruct-q4_K_M (Ollama)
- **모델 우선순위**: llama3.1:8b-instruct-q4_K_M → llama3.1:8b-instruct-q4_1 → gemma3:4b → deepseek-r1:8b → deepseek-r1:14b
- **Claude CLI**: v2.1.29

### Type 분류 로직 (Stage 2)
1. **키워드 매칭** (우선)
   - Type A (실적형): 은행, 보험, 증권, 부동산, 건설, 유통 등
   - Type B (성장형): AI, 반도체, 바이오, 2차전지, 로봇 등
   - 짧은 키워드(2자 이하): 토큰 완전 일치만 허용
   - 긴 키워드(3자 이상): 부분 문자열 매칭 허용

2. **LLM 배치 분석** (키워드 미매칭 시)
   - 불확실한 테마들을 모아서 한 번에 LLM 호출

### 다음 작업 (TODO)
- [ ] 인코딩 수정사항 커밋
- [ ] Stage 3+ 구현/테스트 (종목 선정, 검증)
- [ ] Cloud LLM 검증 단계 검토 (선택적)
- [ ] 전체 파이프라인 통합 테스트

### CLI 사용법
```bash
# 기본 실행
python -m src.orchestrator.pipeline_v2

# 테스트용 (테마 3개, 테마당 종목 2개, Stage 2까지)
python -m src.orchestrator.pipeline_v2 --themes 3 --stocks-per-theme 2 --stage 2 --debug

# 옵션
--themes N           # 최대 테마 수 (기본: 6)
--stocks N           # 최대 종목 수 (기본: 30)
--stocks-per-theme N # 테마당 최대 종목 수
--stage N            # 실행할 최대 스테이지 (0, 1, 2, all)
--skip-preflight     # Preflight 건너뛰기
--debug              # 상세 데이터 흐름 출력
```
