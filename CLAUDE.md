# zuda — Claude Code Orchestrator

> **zuda**: Zig Unofficial Datastructures and Algorithms
> Current Phase: **Phase 1 — Foundations**

---

## Project Overview

- **Language**: Zig 0.15.x
- **Type**: Library (consumed via `build.zig.zon`)
- **Build**: `zig build` / `zig build test`
- **PRD**: `docs/PRD.md` (전체 요구사항 참조)
- **Branch Strategy**: `main` (development)

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
│   └── internal/                #   Shared utilities (not public API)
│       ├── testing.zig          #     Property-based test helpers
│       └── bench.zig            #     Micro-benchmark harness
├── tests/                       # Integration & fuzz tests
├── bench/                       # Benchmark suites
├── examples/                    # Runnable usage examples
└── docs/                        # Documentation
```

---

## Development Workflow

### Autonomous Development Protocol

Claude Code는 이 프로젝트에서 **완전 자율 개발**을 수행한다. 다음 프로토콜을 따른다:

1. **작업 수신** → PRD 또는 사용자 지시를 분석
2. **계획 수립** → 대화형 세션: `EnterPlanMode`로 사용자 승인; 자율 세션(`claude -p`): 내부적으로 계획 후 즉시 구현 진행 (plan mode 도구 사용 금지)
3. **팀 구성** → 작업 복잡도에 따라 동적으로 팀/서브에이전트 생성
4. **구현** → 코딩, 테스트, 리뷰를 병렬 수행
5. **검증** → `zig build test`로 전체 테스트 통과 확인
6. **커밋** → 변경사항 커밋 (사용자 요청 시)
7. **메모리 갱신** → `.claude/memory/`에 기록

### Team Orchestration

복잡한 작업 시 다음 패턴으로 팀을 구성한다:

```
Leader (orchestrator)
├── zig-developer   — 구현 담당
├── code-reviewer   — 코드 리뷰 & 품질 보증
├── test-writer     — 테스트 작성
└── architect       — 설계 검토 (필요 시)
```

**팀 생성 기준**:
- 3개 이상 파일 수정이 필요한 작업 → 팀 구성
- 단일 파일 수정 → 직접 수행
- 아키텍처 변경 → architect 포함

**팀 해산**: 작업 완료 후 반드시 `shutdown_request` → `TeamDelete`로 정리

### Automated Session Execution

자동화 세션(cron job 등)에서는 다음 프로토콜을 순서대로 실행한다.

**컨텍스트 복원** — 세션 시작 시 다음 파일을 읽어 프로젝트 상태 파악:
1. `.claude/memory/project-context.md` — 현재 phase, 체크리스트, 진행 상황
2. `.claude/memory/architecture.md` — 아키텍처 결정사항
3. `.claude/memory/decisions.md` — 기술 결정 로그
4. `.claude/memory/debugging.md` — 알려진 이슈와 해결법
5. `.claude/memory/patterns.md` — 검증된 코드 패턴

**9단계 실행 사이클**:

| Phase | 내용 | 비고 |
|-------|------|------|
| 1. 상태 파악 | `/status` 실행, git log·빌드·테스트 상태 점검 | 체크리스트에서 다음 미완료 항목 식별 |
| 1.5. 이슈 확인 | `gh issue list --state open --limit 10` | 아래 **이슈 우선순위 프로토콜** 참조 |
| 2. 계획 | 구현 전략을 내부적으로 수립 (텍스트 출력) | `EnterPlanMode`/`ExitPlanMode` 사용 금지 — 비대화형 세션에서 블로킹됨 |
| 3. 구현 → 검증 → 커밋 (반복) | 아래 **구현 루프** 참조 | 단위별로 즉시 커밋+푸시 |
| 4. 코드 리뷰 | `/review` — PRD 준수·메모리 안전성·테스트 커버리지 확인 | 이슈 발견 시 수정 후 재커밋 |
| 5. 릴리즈 판단 | 마일스톤 완료 또는 버그 수정 시 **자동 릴리즈** | 아래 **릴리즈 판단 프로토콜** 참조 |
| 6. 메모리 갱신 | `.claude/memory/` 파일 업데이트 | 별도 커밋: `chore: update session memory` → push |
| 7. 세션 요약 | 구조화된 요약 출력 | 아래 템플릿 참조 |

**구현 루프** (Phase 3 상세):

작업을 작은 단위로 분할하고, 각 단위마다 다음을 반복한다:
1. 코드 작성 (하나의 모듈/파일 단위)
2. 테스트 작성 및 `zig build test` 통과 확인
3. 즉시 커밋 + `git push` — 다음 단위로 넘어가기 전에 반드시 수행
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
- phase 완료 (`[x]` 모두 체크) → **MINOR** (v0.X.0)
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
- **Testing**: Every public function must have corresponding tests in the same file
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

Every container follows this structural pattern:

```zig
pub fn RedBlackTree(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime compareFn: fn (ctx: Context, a: K, b: K) std.math.Order,
) type {
    return struct {
        const Self = @This();
        pub const Entry = struct { key: K, value: V };
        pub const Iterator = struct { ... };

        // -- Lifecycle --
        pub fn init(allocator: std.mem.Allocator) Self { ... }
        pub fn deinit(self: *Self) void { ... }
        pub fn clone(self: *const Self) !Self { ... }

        // -- Capacity --
        pub fn count(self: *const Self) usize { ... }
        pub fn isEmpty(self: *const Self) bool { ... }

        // -- Modification --
        /// Time: O(log n) | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V { ... }
        /// Time: O(log n) | Space: O(1)
        pub fn remove(self: *Self, key: K) ?Entry { ... }

        // -- Lookup --
        /// Time: O(log n) | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V { ... }
        pub fn contains(self: *const Self, key: K) bool { ... }

        // -- Iteration --
        pub fn iterator(self: *const Self) Iterator { ... }

        // -- Debug --
        pub fn format(self: *const Self, ...) !void { ... }
        pub fn validate(self: *const Self) !void { ... }
    };
}
```

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

---

## External Tool: zr (Task Runner)

zuda는 **zr** (Zig Task Runner)을 빌드/테스트/CI 워크플로우 자동화에 사용한다.

- **설정 파일**: `zr.toml` (프로젝트 루트)
- **바이너리**: `/Users/fn/Desktop/codespace/zr/zig-out/bin/zr`
- **주요 명령어**: `zr run build`, `zr run test`, `zr run check`, `zr workflow ci`

### zr 이슈 발행 프로토콜

zr 사용 중 버그를 발견하거나 필요한 기능이 없을 때 GitHub Issue를 발행한다.

**버그 발행**:
```bash
gh issue create --repo yusa-imit/zr \
  --title "bug: <간단한 설명>" \
  --label "bug,from:zuda" \
  --body "## 증상
<어떤 문제가 발생했는지>

## 재현 방법
<zr.toml 설정 또는 명령어>

## 기대 동작
<어떻게 동작해야 하는지>

## 환경
- zr: $(zr version)
- zig: $(zig version)
- OS: $(uname -s) $(uname -m)"
```

**기능 요청 발행**:
```bash
gh issue create --repo yusa-imit/zr \
  --title "feat: <필요한 기능>" \
  --label "feature-request,from:zuda" \
  --body "## 필요한 이유
<zuda에서 왜 이 기능이 필요한지>

## 제안하는 설정/사용법
<원하는 zr.toml 설정이나 CLI 사용 예시>

## 대안
<현재 워크어라운드가 있다면 설명>"
```

**발행 조건**:
- zr의 기존 기능으로 해결할 수 없는 문제일 때만 발행
- 동일한 이슈가 이미 열려있는지 먼저 확인: `gh issue list --repo yusa-imit/zr --state open --search "<keyword>"`
- 이슈 발행 후 현재 작업으로 복귀 (zr 수정을 직접 하지 않음)
- zr 에이전트(cron job)가 `from:*` 라벨 이슈를 최우선으로 처리한다

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
# 소비자 프로젝트에 마이그레이션 가능 알림
gh issue create --repo yusa-imit/<consumer> \
  --title "feat: migrate to zuda for <data-structure>" \
  --label "feature-request,from:zuda" \
  --body "## zuda 대체 가능 알림

zuda에서 \`<DataStructure>\`가 구현 완료되었습니다.

## 현재 자체 구현
- 파일: \`<consumer-file-path>\`
- LOC: <lines>

## zuda API
\`\`\`zig
const ds = @import(\"zuda\").containers.<category>.<DataStructure>;
\`\`\`

## 마이그레이션 가이드
1. \`build.zig.zon\`에 zuda 의존성 추가
2. 자체 구현을 zuda import로 교체
3. API 차이점: <간략 설명>

## 테스트
- zuda 테스트: <통과 수> passing
- 벤치마크: <성능 수치>"
```

---

## Phase Implementation Roadmap

### Phase 1 — Foundations (Weeks 1–8)
- [ ] Project scaffolding: CI, testing harness, benchmark framework
- [ ] **Lists & Queues**: `SkipList`, `XorLinkedList`, `UnrolledLinkedList`, `Deque`
- [ ] **Hash containers**: `CuckooHashMap`, `RobinHoodHashMap`, `SwissTable`, `ConsistentHashRing`
- [ ] **Heaps**: `FibonacciHeap`, `BinomialHeap`, `PairingHeap`, `DaryHeap`
- [ ] All Phase 1 containers pass invariant tests, fuzz tests (1hr minimum), and benchmarks

### Phase 2 — Trees & Range Queries (Weeks 9–16)
- [ ] **Balanced BSTs**: `RedBlackTree`, `AVLTree`, `SplayTree`, `AATree`, `ScapegoatTree`
- [ ] **Tries & B-Trees**: `Trie`, `RadixTree`, `BTree`
- [ ] **Range query**: `SegmentTree`, `LazySegmentTree`, `FenwickTree`, `SparseTable`, `IntervalTree`
- [ ] **Spatial**: `KDTree`, `RTree`, `QuadTree`, `OctTree`
- [ ] **Strings**: `SuffixArray`, `SuffixTree`

### Phase 3 — Graph Algorithms (Weeks 17–24)
- [ ] **Representations**: `AdjacencyList`, `AdjacencyMatrix`, `CompressedSparseRow`, `EdgeList`
- [ ] **Traversal & shortest paths**: BFS, DFS, Dijkstra, Bellman-Ford, A*, Floyd-Warshall, Johnson's
- [ ] **MST & connectivity**: Kruskal, Prim, Borůvka, Tarjan SCC, Kosaraju, bridges, articulation points
- [ ] **Flow & matching**: Edmonds-Karp, Dinic, Push-Relabel, Hopcroft-Karp, Hungarian, topological sort

### Phase 4 — Algorithms & Probabilistic (Weeks 25–34)
- [ ] **Sorting**: TimSort, IntroSort, RadixSort, CountingSort, BlockSort, in-place MergeSort
- [ ] **String algorithms**: KMP, Boyer-Moore, Rabin-Karp, Aho-Corasick, Z-algorithm
- [ ] **Probabilistic & cache**: `BloomFilter`, `CuckooFilter`, `CountMinSketch`, `HyperLogLog`, `LRUCache`, `LFUCache`
- [ ] **Math & geometry**: GCD, modexp, Miller-Rabin, convex hull, closest pair
- [ ] **DP utilities**: LIS, LCS, edit distance, knapsack, binary search variants

### Phase 5 — Advanced & Polish (Weeks 35–44)
- [ ] **Concurrent**: `LockFreeQueue`, `LockFreeStack`, `ConcurrentSkipList`, `ConcurrentHashMap`
- [ ] **Persistent**: `PersistentArray`, `PersistentRBTree`, `PersistentHashMap` (HAMT)
- [ ] **Exotic**: `DisjointSet`, `VanEmdeBoasTree`, `DancingLinks`, `Rope`, `BK-Tree`
- [ ] **C API & FFI**: C header generation, binding examples
- [ ] **Documentation & v1.0**: API reference, algorithm explainers, decision-tree guide

---

## Performance Targets

| Metric | Target |
|--------|--------|
| RedBlackTree insert | ≤ 200 ns/op (1M random keys) |
| RedBlackTree lookup | ≤ 150 ns/op (1M random keys) |
| BTree(128) range scan | ≥ 50M keys/sec (sequential) |
| FibonacciHeap decrease-key | ≤ 50 ns amortized |
| BloomFilter lookup | ≥ 100M ops/sec |
| Dijkstra (1M nodes, 5M edges) | ≤ 500 ms |
| TimSort (1M i64, random) | ≤ 10% overhead vs `std.sort` |
| Aho-Corasick (1000 patterns, 1MB text) | ≥ 500 MB/sec |

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
