# zuda — Claude Code Orchestrator

> **zuda**: Zig Universal Datastructures and Algorithms
> Current Phase: **v1.x Complete — v2.0 Scientific Computing Track**

---

## Project Overview

- **Language**: Zig 0.15.x
- **Type**: Library (consumed via `build.zig.zon`)
- **Build**: `zig build` / `zig build test`
- **PRD**: `docs/PRD.md` (전체 요구사항 참조)
- **Branch Strategy**: `main` (development)
- **Vision**: v1.x는 100+ 자료구조 & 80+ 알고리즘 라이브러리 (완료). v2.0은 선형대수, 통계, FFT, 수치해석, 최적화를 포함하는 **과학 컴퓨팅 플랫폼**으로 확장 — NumPy/SciPy의 Zig-native 대안
- **v2.0 Modules**: `ndarray/` (NDArray), `linalg/` (선형대수), `stats/` (통계), `signal/` (FFT/신호처리), `numeric/` (수치해석), `optimize/` (최적화)

## Repository Structure

```
zuda/
├── CLAUDE.md                    # THIS FILE — orchestrator
├── docs/PRD.md                  # Product Requirements Document
├── .gitignore                   # Git ignore rules
├── .claude/
│   ├── agents/                  # Custom subagent definitions
│   │   ├── zig-developer.md     #   model: haiku  — Zig 구현
│   │   ├── code-reviewer.md     #   model: sonnet — 코드 리뷰
│   │   ├── test-writer.md       #   model: haiku  — 테스트 작성
│   │   ├── architect.md         #   model: sonnet — 아키텍처 설계
│   │   ├── git-manager.md       #   model: haiku  — Git 운영
│   │   └── ci-cd.md             #   model: haiku  — CI/CD 관리
│   ├── commands/                # Slash commands
│   ├── memory/                  # Persistent agent memory
│   └── settings.json            # Claude Code permissions
├── .github/workflows/           # CI/CD pipelines
│   └── ci.yml                   #   Build, test, cross-compile
├── src/
│   ├── root.zig                 #   Library root — re-exports all public types
│   ├── main.zig                 #   Executable entry point
│   ├── containers/              #   Data Structures
│   │   ├── lists/               #     Sequential containers
│   │   ├── trees/               #     Tree-based containers
│   │   ├── graphs/              #     Graph representations
│   │   ├── heaps/               #     Heap variants
│   │   ├── hashing/             #     Hash-based containers
│   │   ├── queues/              #     Queue / deque variants
│   │   ├── strings/             #     String-specialized structures
│   │   ├── spatial/             #     Spatial index structures
│   │   └── probabilistic/       #     Bloom filter, Count-Min Sketch, etc.
│   ├── algorithms/              #   Algorithms
│   │   ├── sorting/
│   │   ├── searching/
│   │   ├── graph/
│   │   ├── string/
│   │   ├── math/
│   │   ├── geometry/
│   │   └── dynamic_programming/
│   ├── iterators/               #   Composable iterator adaptors
│   ├── ndarray/                 #   N-dimensional Array (v2.0)
│   ├── linalg/                  #   Linear Algebra — BLAS, decompositions, solvers (v2.0)
│   ├── stats/                   #   Statistics — distributions, testing, regression (v2.0)
│   ├── signal/                  #   Signal Processing — FFT, convolution, filtering (v2.0)
│   ├── numeric/                 #   Numerical Methods — integration, ODE, interpolation (v2.0)
│   └── internal/                #   Shared utilities (not public API)
│       ├── testing.zig          #     Property-based test helpers
│       └── bench.zig            #     Micro-benchmark harness
├── tests/                       # Integration & fuzz tests
├── bench/                       # Benchmark suites
├── examples/                    # Runnable usage examples
└── docs/                        # Documentation
```

> **NOTE**: 위 디렉토리 구조와 PRD의 구조는 **참고용**이다. 실제 구현 과정에서 폴더 구조, 파일명, 모듈 구성 등은 필요에 따라 변경될 수 있다.

---

## Development Workflow

### Autonomous Development Protocol

Claude Code는 이 프로젝트에서 **완전 자율 개발**을 수행한다. 다음 프로토콜을 따른다:

1. **작업 수신** → PRD 또는 사용자 지시를 분석
2. **계획 수립** → 대화형 세션: `EnterPlanMode`로 사용자 승인; 자율 세션(`claude -p`): 내부적으로 계획 후 즉시 구현 진행 (plan mode 도구 사용 금지)
3. **팀 구성** → 작업 복잡도에 따라 동적으로 팀/서브에이전트 생성
4. **구현** → TDD 사이클: 테스트 작성(test-writer) → 구현(zig-developer) → 리뷰 순차 수행
5. **검증** → `zig build test`로 전체 테스트 통과 확인
6. **커밋** → 변경사항 커밋 (사용자 요청 시)
7. **메모리 갱신** → `.claude/memory/`에 기록

### Team Orchestration

복잡한 작업 시 다음 패턴으로 팀을 구성한다:

```
Leader (orchestrator)
├── test-writer     — 테스트 먼저 작성 (MUST run before zig-developer)
├── zig-developer   — 테스트를 통과시키는 구현
├── code-reviewer   — 코드 리뷰 & 품질 보증
└── architect       — 설계 검토 (필요 시)
```

**TDD 실행 규칙**:
- `test-writer`는 모든 구현 작업에서 필수로 먼저 호출한다 (단일 파일 수정 포함)
- `zig-developer`는 `test-writer`가 작성한 실패하는 테스트가 존재한 후에만 호출한다
- 테스트 수정이 필요하면 `zig-developer`가 직접 수정하지 않고 `test-writer`를 재호출한다
- 테스트는 커버리지 수치가 아닌 의미 있는 검증을 기준으로 작성한다

**팀 생성 기준**:
- 3개 이상 파일 수정 → 팀 구성 (test-writer 필수 포함)
- 단일 파일 수정 → test-writer 서브에이전트 호출 후 직접 구현
- 아키텍처 변경 → architect + test-writer 포함

**팀 해산**: 작업 완료 후 반드시 `shutdown_request` → `TeamDelete`로 정리

### Automated Session Execution

자동화 세션(cron job 등)에서는 다음 프로토콜을 순서대로 실행한다.

**컨텍스트 복원** — 세션 시작 시 다음 파일을 읽어 프로젝트 상태 파악:
1. `.claude/memory/project-context.md` — 현재 phase, 체크리스트, 진행 상황
2. `.claude/memory/architecture.md` — 아키텍처 결정사항
3. `.claude/memory/decisions.md` — 기술 결정 로그
4. `.claude/memory/debugging.md` — 알려진 이슈와 해결법
5. `.claude/memory/patterns.md` — 검증된 코드 패턴
6. `docs/milestones.md` — 마일스톤 로드맵, 진행 상황, 성능 목표

**9단계 실행 사이클**:

| Phase | 내용 | 비고 |
|-------|------|------|
| 1. 상태 파악 | `/status` 실행, git log·빌드·테스트 상태 점검 | `docs/milestones.md`에서 다음 작업 식별 |
| 1.5. 이슈 확인 | `gh issue list --state open --limit 10` | 아래 **이슈 우선순위 프로토콜** 참조 |
| 2. 계획 | 구현 전략을 내부적으로 수립 (텍스트 출력) | `EnterPlanMode`/`ExitPlanMode` 사용 금지 — 비대화형 세션에서 블로킹됨 |
| 3. 구현 → 검증 → 커밋 (반복) | 아래 **구현 루프** 참조 | 단위별로 즉시 커밋+푸시 |
| 4. 코드 리뷰 | `/review` — PRD 준수·메모리 안전성·테스트 커버리지 확인 | 이슈 발견 시 수정 후 재커밋 |
| 5. 릴리즈 판단 | 마일스톤 완료 또는 버그 수정 시 **자동 릴리즈** | 아래 **릴리즈 판단 프로토콜** 참조 |
| 6. 메모리 갱신 | `.claude/memory/` 및 `docs/milestones.md` 업데이트 | 별도 커밋: `chore: update session memory` → push |
| 7. 세션 요약 | 구조화된 요약 출력 | 아래 템플릿 참조 |

**구현 루프** (Phase 3 상세):

작업을 작은 단위로 분할하고, 각 단위마다 다음을 반복한다:
0. **Scratchpad 초기화** — `.claude/scratchpad.md`를 초기화 템플릿으로 덮어쓰기 (Shared Scratchpad Protocol 참조)
1. **Red** — `test-writer` 호출: 요구사항을 검증하는 실패하는 테스트 작성
2. **Green** — `zig-developer` 호출: 테스트를 통과시키는 최소한의 구현
3. **Refactor** — 테스트 통과 상태에서 코드 정리 (테스트 수정 필요 시 `test-writer` 재호출)
4. 즉시 커밋 + `git push` — 다음 단위로 넘어가기 전에 반드시 수행
- 미커밋 변경사항을 여러 파일에 걸쳐 누적하지 않는다
- 한 사이클 내에 완료할 수 없는 작업은 동작하는 중간 상태로 커밋+푸시한다
- `git add -A` 금지 — 변경된 파일을 명시적으로 지정

**이슈 우선순위 프로토콜** (Phase 1.5):

세션 시작 시 GitHub Issues를 확인하고, PRD 기능과 비교하여 우선순위를 결정한다:

```bash
gh issue list --state open --limit 10 --json number,title,labels,createdAt
```

**우선순위 판단 기준**:

| 우선순위 | 조건 | 예시 |
|---------|------|------|
| 1 (최우선) | `bug` 라벨 + 정확성/안전성 관련 | 데이터 구조 불변조건 위반, 메모리 누수 |
| 2 (높음) | `bug` 라벨 (일반) | 테스트 실패, 빌드 오류 |
| 3 (보통) | `feature-request` 라벨 + 현재 phase 범위 내 | 현재 구현 중인 phase의 추가 기능 |
| 4 (낮음) | `feature-request` 라벨 + 미래 phase | 아직 시작하지 않은 phase의 기능 요청 |

**판단 규칙**:
- 우선순위 1-2 (버그): PRD 작업보다 **항상 우선** 처리
- 우선순위 3 (현재 phase 기능 요청): PRD 작업과 **병행** — 같은 모듈 작업 시 함께 구현
- 우선순위 4 (미래 phase 기능 요청): **적어두고 넘어감** — 해당 phase 도달 시 처리
- 이슈를 처리한 후: `gh issue close <number> --comment "Fixed in <commit-hash>"`

**테스트 품질 감사** (Stability 세션 필수):
- 무조건 통과하는 무의미한 테스트 식별 및 개선 (예: 빈 assertion, 항상 true인 조건)
- 구현 코드를 그대로 복사한 expected value 제거
- happy-path-only 테스트에 실패 시나리오 보강
- 경계값, 에러 경로, 동시성 시나리오 누락 확인
- `test-writer`를 호출하여 개선 방향 수립

### 버전 안전 규칙 (CRITICAL)

- **버전은 반드시 단조 증가**해야 한다. 새 버전은 `build.zig.zon`의 현재 버전보다 **반드시 높아야** 한다.
- 릴리즈 전 반드시 현재 버전을 확인: `grep 'version' build.zig.zon`
- 새 태그가 `git tag -l 'v*' --sort=-v:refname | head -1`보다 **낮으면 즉시 중단**.
- 버전 다운그레이드는 **절대 금지**.
- **버전 건너뛰기 금지**: 릴리즈 버전은 현재 `build.zig.zon` 버전의 **다음 마이너**여야 한다. 마일스톤에 미리 할당된 버전 번호가 있더라도, 실제 릴리즈 시점에는 현재 버전 + 1을 사용한다.
- **마일스톤은 이름(테마)으로 관리**: 버전 번호는 릴리즈 시점에 결정. 마일스톤 수립 시 미리 할당된 번호는 참고용.

**릴리즈 판단 프로토콜** (Step 5):

```bash
# 릴리즈 판단 체크
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
git log ${LAST_TAG}..HEAD --oneline
gh issue list --state open --label bug --limit 5
```

**판단 로직**:
- 태그 이후 커밋 없음 → **SKIP** (릴리즈 불필요)
- `fix:` 커밋만 존재 → **PATCH** (v0.X.Y)
- phase 완료 (`docs/milestones.md`에서 `[x]` 모두 체크) → **MINOR** (v0.X.0)
- **MAJOR** (v1.0.0) → 사용자 지시 시에만 수행

**공통 릴리즈 조건 (ALL must be true)**:
1. `zig build test` — 전체 통과, 0 failures
2. 6개 크로스 컴파일 타겟 빌드 성공
3. `bug` 라벨 이슈가 **0개** (open)

---

**세션 요약 템플릿**:

    ## Session Summary
    ### Completed
    - [이번 사이클에서 완료한 내용]
    ### Files Changed
    - [생성/수정된 파일 목록]
    ### Tests
    - [테스트 수, 통과/실패 상태]
    ### Benchmarks
    - [성능 목표 대비 현황 (해당 시)]
    ### Next Priority
    - [다음 사이클에서 작업할 내용]
    ### Issues / Blockers
    - [발생한 문제 또는 미해결 이슈]

### Available Custom Agents

| Agent | Model | File | Purpose |
|-------|-------|------|---------|
| zig-developer | haiku | `.claude/agents/zig-developer.md` | Zig 코드 구현, 빌드 오류 해결 |
| code-reviewer | sonnet | `.claude/agents/code-reviewer.md` | 코드 리뷰, 품질/보안 검사 |
| test-writer | haiku | `.claude/agents/test-writer.md` | 유닛/프로퍼티/퍼즈 테스트 작성 |
| architect | sonnet | `.claude/agents/architect.md` | 아키텍처 설계, 모듈 구조 결정 |
| git-manager | haiku | `.claude/agents/git-manager.md` | Git 운영, 브랜치/커밋 관리 |
| ci-cd | haiku | `.claude/agents/ci-cd.md` | GitHub Actions, CI/CD 파이프라인 |

### Available Slash Commands

| Command | File | Purpose |
|---------|------|---------|
| /build | `.claude/commands/build.md` | 라이브러리 빌드 |
| /test | `.claude/commands/test.md` | 테스트 실행 |
| /review | `.claude/commands/review.md` | 현재 변경사항 코드 리뷰 |
| /implement | `.claude/commands/implement.md` | 기능 구현 워크플로우 |
| /fix | `.claude/commands/fix.md` | 버그 수정 워크플로우 |
| /status | `.claude/commands/status.md` | 프로젝트 상태 확인 |
| /release | `.claude/commands/release.md` | 릴리스 워크플로우 |
| /bench | `.claude/commands/bench.md` | 벤치마크 실행 |

---

## Coding Standards

### Zig Conventions

- **Naming**: camelCase for functions/variables, PascalCase for types, SCREAMING_SNAKE for constants
- **Error handling**: Always use explicit error unions, never `catch unreachable` in library code
- **Memory**: Every container accepts `std.mem.Allocator` — never hardcode allocator. Provide Managed and Unmanaged variants.
- **Testing**: 모든 공개 함수는 구현 전에 실패하는 테스트를 먼저 작성한다 (TDD). 테스트는 커버리지가 아닌 실제 동작 검증에 집중한다
- **Comments**: Only where logic is non-obvious. All public functions must have doc comments with Big-O complexity.
- **Imports**: Group stdlib, then project imports, then test imports

### Library-Specific Rules

- **Allocator-first** — Every heap-allocating container takes `std.mem.Allocator`
- **Comptime configuration** — Parameterize behavior (comparator, hash, branching factor) at compile time
- **Iterator protocol** — All iterable containers expose `next() -> ?T`
- **Complexity contracts** — Every public function documents Big-O time and space in doc comments
- **Fixed-capacity variants** — Provide `Bounded*` variants for embedded/latency-sensitive contexts
- **No `@panic`** — Return errors, let caller decide
- **No `std.debug.print`** — Use proper writer-based output for debug formatting
- **`validate()` invariant checks** — Every container must have a `validate()` method asserting internal invariants

### File Organization

- One data structure per file (e.g., `src/containers/trees/red_black_tree.zig`)
- Keep files under 800 lines; split into submodules if exceeded
- Public API at top of file, private helpers at bottom
- Tests at the bottom of each file within `test` block
- Container files follow pattern: type definition → lifecycle (init/deinit) → modification → lookup → iteration → bulk → debug/validate → tests

### Error Messages

Library errors should be descriptive:
```zig
error.KeyNotFound      // not error.NotFound
error.TreeInvariant    // not error.Invalid
error.CapacityExceeded // not error.Full
error.CycleDetected    // not error.Cycle
```

### Generic Container Template

Every container follows this structural pattern: `pub fn Container(comptime K, V, Context, compareFn) type` returning a struct with sections in order:
1. **Lifecycle**: `init(allocator)`, `deinit()`, `clone()`
2. **Capacity**: `count()`, `isEmpty()`
3. **Modification**: `insert()`, `remove()` — with `/// Time: O(...) | Space: O(...)` doc comments
4. **Lookup**: `get()`, `contains()`
5. **Iteration**: `iterator() -> Iterator` (exposes `next() -> ?T`)
6. **Debug**: `format()`, `validate()` (invariant assertions)

---

## Git Workflow

### Branch Strategy

- `main` — primary development branch
- Feature branches: `feat/<name>`, `fix/<name>`, `refactor/<name>`

### Commit Convention

```
<type>: <subject>

<body>

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `perf`, `ci`

---

## Memory System

### Long-Term Memory Preservation

에이전트와 오케스트레이터는 `.claude/memory/` 디렉토리에 장기 기억을 보존한다.

**메모리 파일 구조**:
```
.claude/memory/
├── project-context.md    # 프로젝트 개요, 현재 phase, 체크리스트
├── architecture.md       # 아키텍처 결정사항
├── decisions.md          # 주요 기술 결정 로그
├── debugging.md          # 디버깅 인사이트, 해결된 문제
└── patterns.md           # 검증된 코드 패턴
```

**메모리 프로토콜**:
1. 세션 시작 시 `.claude/memory/` 파일들을 읽어 컨텍스트 복원
2. 중요한 결정/발견 시 즉시 해당 메모리 파일에 기록
3. 메모리 파일이 200줄을 초과하면 핵심만 남기고 압축

### Shared Scratchpad Protocol

개발 사이클(Red-Green-Refactor) 중 서브에이전트 간 협업을 위한 **임시 공유 메모리**이다.
영구 메모리(`.claude/memory/`)와 독립 운영되며, 기존 메모리 업데이트 규칙은 변경되지 않는다.

**파일**: `.claude/scratchpad.md` — `.gitignore`에 등록, git에 커밋하지 않는다

**대상 에이전트**: `test-writer`, `zig-developer`, `code-reviewer`

**라이프사이클**:
1. **사이클 시작** — 오케스트레이터가 `.claude/scratchpad.md`를 초기화 (기존 내용 덮어쓰기)
2. **에이전트 작업** — 각 에이전트가 작업 전 로드 → 작업 후 기록
3. **사이클 종료** — 다음 사이클 시작 시 다시 초기화

**규칙**:
1. **MUST LOAD**: 대상 에이전트는 작업 시작 시 `.claude/scratchpad.md`를 **반드시** 읽는다
2. **MUST WRITE**: 작업 완료 후 자신의 작업 내용을 **반드시** 추가한다
3. **NO DELETE**: 다른 에이전트의 기록을 삭제하지 않는다 (append-only)
4. **EPHEMERAL**: git에 커밋하지 않는다 — 사이클 내 협업이 목적
5. **NOT MEMORY**: 영구 보존이 필요한 인사이트는 `.claude/memory/`에 별도 기록 (기존 규칙 준수)

**초기화 템플릿** — 오케스트레이터가 사이클 시작 시 작성:

```markdown
# Scratchpad — [작업 설명]
> Cycle started: [timestamp]
> Goal: [이번 사이클의 목표]
---
```

**에이전트 기록 형식** — 작업 완료 후 append:

```markdown
## [agent-name] — [timestamp]
- **Did**: [수행한 작업]
- **Why**: [근거 / 의도]
- **Files**: [변경한 파일 목록]
- **For next**: [다음 에이전트가 알아야 할 사항]
- **Issues**: [발견한 문제점, 없으면 생략]
```

---

## External Tool: zr (Task Runner)

zuda는 **zr** (Zig Task Runner)을 빌드/테스트/CI 워크플로우 자동화에 사용한다.

- **설정 파일**: `zr.toml` (프로젝트 루트)
- **바이너리**: `/Users/fn/Desktop/codespace/zr/zig-out/bin/zr`
- **주요 명령어**: `zr run build`, `zr run test`, `zr run check`, `zr workflow ci`

### zr 이슈 발행 프로토콜

zr 사용 중 버그 발견 또는 기능 부재 시 GitHub Issue를 발행한다.

- **버그**: `gh issue create --repo yusa-imit/zr --title "bug: <설명>" --label "bug,from:zuda"` — body에 증상, 재현 방법, 기대 동작, 환경 포함
- **기능 요청**: `gh issue create --repo yusa-imit/zr --title "feat: <기능>" --label "feature-request,from:zuda"` — body에 필요 이유, 제안 사용법, 대안 포함
- 발행 전 중복 확인: `gh issue list --repo yusa-imit/zr --state open --search "<keyword>"`
- 이슈 발행 후 현재 작업으로 복귀 (zr 수정을 직접 하지 않음)

---

## Consumer Use Case Registry

zuda의 설계와 우선순위는 실제 소비자 프로젝트의 사용례에 기반한다. 아래는 각 프로젝트에서 직접 구현하여 사용 중인 자료구조/알고리즘이며, zuda v1.0 이전에 모두 대체 가능해야 한다.

### zr (Task Runner) — `../zr`

| 현재 구현 | 파일 | LOC | zuda 대체 | Phase |
|-----------|------|-----|-----------|-------|
| DAG (Directed Acyclic Graph) | `src/graph/dag.zig` | 187 | `AdjacencyList` + graph utils | 3 |
| Topological Sort (Kahn's) | `src/graph/topo_sort.zig` | 323 | `algorithms/graph/topological_sort` | 3 |
| Cycle Detection | `src/graph/cycle_detect.zig` | 205 | `algorithms/graph/cycle_detection` | 3 |
| Work-Stealing Deque (Chase-Lev) | `src/exec/workstealing.zig` | 130 | `StealingQueue` | 1 |
| Levenshtein Distance | `src/util/levenshtein.zig` | 214 | `algorithms/dynamic_programming/edit_distance` | 4 |
| String Pool / Interning | `src/util/string_pool.zig` | 97 | (domain-specific, 참고만) | — |
| Glob Pattern Matching | `src/util/glob.zig` | 130 | `algorithms/string/glob_match` | 4 |
| Object Pool | `src/util/object_pool.zig` | 130 | (domain-specific, 참고만) | — |
| ASCII Graph Visualization | `src/graph/ascii.zig` | 170 | (domain-specific, 참고만) | — |

### silica (Embedded RDBMS) — `../silica`

| 현재 구현 | 파일 | LOC | zuda 대체 | Phase |
|-----------|------|-----|-----------|-------|
| B+Tree | `src/storage/btree.zig` | 4300 | `BTree` | 2 |
| Buffer Pool (LRU Cache) | `src/storage/buffer_pool.zig` | 1237 | `LRUCache` | 4 |
| Lock Manager + Deadlock Detection (DFS) | `src/tx/lock.zig` | 1463 | Graph cycle detection | 3 |
| Free Space Map | `src/storage/fsm.zig` | 709 | (domain-specific, 참고만) | — |
| Varint (LEB128) | `src/util/varint.zig` | 284 | (domain-specific, 참고만) | — |
| CRC32C Checksum | `src/util/checksum.zig` | 121 | (domain-specific, 참고만) | — |
| B+Tree Cursor (Iterator) | `src/storage/btree.zig` | 230 | Iterator protocol 참고 | 2 |

### zoltraak (Redis-Compatible Server) — `../zoltraak`

| 현재 구현 | 파일 | LOC | zuda 대체 | Phase |
|-----------|------|-----|-----------|-------|
| Sorted Set (HashMap + sorted ArrayList) | `src/storage/memory.zig` | 1800 | `SkipList` 또는 `RedBlackTree` | 1/2 |
| HyperLogLog | `src/storage/memory.zig` | 80 | `HyperLogLog` | 4 |
| Geohash encoding/decoding | `src/commands/geo.zig` | 1400 | `algorithms/geometry/geohash` | 4 |
| Glob Pattern Matching | `src/utils/glob.zig` | 90 | `algorithms/string/glob_match` | 4 |
| Haversine Distance | `src/commands/geo.zig` | 15 | `algorithms/geometry/haversine` | 4 |
| LRU-style Expiry (lazy + active) | `src/storage/memory.zig` | 50 | `LRUCache` / `LFUCache` | 4 |

### sailor (TUI Framework) — `../sailor`

| 현재 구현 | 파일 | LOC | zuda 대체 | Phase |
|-----------|------|-----|-----------|-------|
| Cell Buffer Diff Engine | `src/tui/buffer.zig` | 90 | (domain-specific, 참고만) | — |
| Layout Constraint Solver | `src/tui/layout.zig` | 196 | (domain-specific, 참고만) | — |
| Grid Layout Algorithm | `src/tui/grid.zig` | 262 | (domain-specific, 참고만) | — |
| Unicode Width Calculator | `src/unicode.zig` | 166 | (domain-specific, 참고만) | — |

> sailor의 구현은 대부분 TUI에 특화되어 zuda로 대체하기보다는 패턴 참고용으로 활용한다.

### 개발 시 참고 프로토콜

zuda에서 데이터 구조/알고리즘을 구현할 때 다음을 따른다:

1. **사용례 먼저 확인** — 위 레지스트리에서 해당 구조의 실제 사용례를 확인
2. **소비자 코드 읽기** — 해당 파일을 `Read`로 읽어 API 패턴, 엣지 케이스, 성능 요구사항 파악
3. **API 호환성 고려** — 소비자가 최소한의 변경으로 zuda로 전환할 수 있는 API 설계
4. **테스트 케이스 참고** — 소비자 프로젝트의 테스트에서 실제 사용 패턴 추출
5. **성능 기준 설정** — 소비자의 현재 구현 대비 동등 이상의 성능 보장

### 소비자 마이그레이션 발행 프로토콜

zuda에서 소비자 프로젝트의 자료구조를 대체할 수 있는 구현이 완료되면:

```bash
gh issue create --repo yusa-imit/<consumer> \
  --title "chore: migrate to zuda for <data-structure>" \
  --label "migration,from:zuda" \
  --body "zuda에서 <DataStructure> 구현 완료. 현재 자체 구현: <file> (<LOC> lines). zuda API: @import(\"zuda\").containers.<category>.<DS>. 마이그레이션: build.zig.zon에 의존성 추가 → import 교체. 테스트: <N> passing, 벤치마크: <수치>"
```

---

## Milestones

See [`docs/milestones.md`](docs/milestones.md) for the full implementation roadmap, phase checklists, and performance targets.

**v1.x status**: Phases 1-5 complete (100+ data structures, 80+ algorithms, 746 tests passing)
**v2.0 track**: Phases 6-12 — Scientific computing (NDArray → linalg → stats → signal → numeric → optimize → release)

---

## Quick Reference

```bash
# Build library
zig build

# Test
zig build test

# Run executable
zig build run

# Cross-compile check
zig build -Dtarget=x86_64-linux-gnu

# Clean
rm -rf zig-out .zig-cache
```

---

## Rules for Claude Code

1. **Always read before writing** — 파일 수정 전 반드시 Read로 현재 내용 확인
2. **Test after every change** — 코드 변경 후 `zig build test` 실행
3. **Incremental commits** — 기능 단위로 작은 커밋
4. **Memory updates** — 중요한 발견/결정은 즉시 메모리에 기록
5. **No over-engineering** — 현재 phase에 필요한 것만 구현
6. **PRD is source of truth** — 기능 요구사항은 `docs/PRD.md` 참조
7. **Team cleanup** — 팀 작업 완료 후 반드시 해산
8. **Library mindset** — 모든 컨테이너는 allocator-first, comptime-parameterized
9. **Complexity contracts** — 모든 공개 함수에 Big-O 복잡도 명시
10. **Invariant validation** — 모든 컨테이너에 `validate()` 메서드 구현
11. **Respect CI** — CI 파이프라인 호환성 유지
12. **Never force push** — 파괴적 git 명령어 금지
13. **Agent activity logging** — Subagent/Team 호출 시 반드시 `.claude/logs/agent-activity.jsonl`에 로그 기록 (아래 Agent Activity Logging 섹션 참조)
14. **TDD is mandatory** — 구현 전 반드시 `test-writer`로 실패하는 테스트를 작성. 테스트 수정 시에도 `test-writer` 재호출
15. **Meaningful tests only** — 무조건 통과하는 테스트, 구현을 복사한 테스트, assertion 없는 테스트 금지. 테스트가 실패할 수 있는 조건이 명확해야 한다

---

## Agent Activity Logging

Subagent(Task 도구) 또는 Team(TeamCreate)을 호출할 때마다 `.claude/logs/agent-activity.jsonl`에 로그를 기록한다.

**로그 형식** (JSON Lines — 한 줄에 하나의 JSON 객체):
```json
{"timestamp":"2026-03-14T12:00:00Z","action":"subagent","agent_type":"zig-developer","task":"Implement BTree insert","project":"zuda"}
{"timestamp":"2026-03-14T12:05:00Z","action":"team_create","team_name":"btree-impl","members":["zig-developer","test-writer"],"task":"Implement BTree module","project":"zuda"}
{"timestamp":"2026-03-14T13:00:00Z","action":"team_delete","team_name":"btree-impl","project":"zuda"}
```

**필드**:

| 필드 | 필수 | 설명 |
|------|------|------|
| `timestamp` | ✅ | ISO 8601 형식 (UTC) |
| `action` | ✅ | `subagent` \| `team_create` \| `team_delete` |
| `agent_type` | subagent 시 | 에이전트 타입 (`zig-developer`, `code-reviewer`, `Explore` 등) |
| `team_name` | team 시 | 팀 이름 |
| `members` | team_create 시 | 팀 멤버 이름 배열 |
| `task` | ✅ | 작업 설명 (Task 도구의 description 또는 prompt 요약) |
| `project` | ✅ | 프로젝트 이름 (`zuda`) |

**규칙**:
1. `.claude/logs/` 디렉토리가 없으면 생성
2. 파일에 append (기존 로그 유지)
3. 로그는 git에 커밋+push 필수 — 커밋 메시지: `chore: update agent activity log`
4. 세션 종료 전 미커밋 로그가 있으면 반드시 커밋+push
