<a id="maintaining-these-reference-docs"></a>
# Ведение этих справочных документов

`ENVIRONMENT.md` и `SETTINGS.md` **генерируются из источника**. Они содержат строку «Сгенерировано из `main @ <hash>`» и *будут* смещаться по мере добавления в код регуляторов. Это повторяемая процедура их обновления — вручную или с помощью искусственного интеллекта (например, Клода), выполняющего утомительное извлечение и редактирование таблиц.

<a id="where-the-truth-lives"></a>
## Где живет истина

| Док | Источник истины | Что сканировать |
|---|---|---|
| `ENVIRONMENT.md` | `c/glm.c` (и другие `c/*.c`) | каждый вызов `getenv("...")` — значение по умолчанию и завершающий `/* comment */` |
| `SETTINGS.md` | `c/coli`, `c/openai_server.py` | каждые `add_parser(...)` и `add_argument(...)` |

Ничто другое не определяет их. Если ручки нет на одном из этих мест вызова, она ненастоящая.

<a id="step-1-extract-the-current-state"></a>
## Шаг 1 — извлекаем текущее состояние

Запустите коммит, который вы документируете (используйте `upstream/dev`, а не локальную ветку):

```bash
cd <repo>
git fetch upstream
HASH=$(git rev-parse --short upstream/dev); echo "documenting $HASH"

# Environment variables (defaults + inline comments).
# Scan .h/.cu/.mm too, not just .c: route_trace.h, rans.h, sample.h and
# omp_tune.h own knobs that every engine inherits, and the CUDA/Metal/Vulkan
# backends own their own. Scanning 'c/*.c' alone silently drops them.
# c/tests/ is excluded: test fixtures read their own variables (EXPERT_RAW,
# TMPDIR) that are not part of the engine's user-facing surface.
# getenv_utf8/compat_getenv_utf8 are the Windows-safe wrappers -- COLI_PROMPT is
# read through one, so a plain 'getenv(' grep reports it as removed.
git grep -nE '(compat_)?getenv(_utf8)?\("' upstream/dev -- 'c/*.c' 'c/*.h' 'c/*.cu' 'c/*.mm' ':!c/tests/*'

# CLI settings:
git grep -nE 'add_parser\(|add_argument\(' upstream/dev -- c/coli c/openai_server.py

# Quick sanity: how many distinct env vars exist now?
git grep -hoE '(compat_)?getenv(_utf8)?\("[A-Z0-9_]+"' upstream/dev -- 'c/*.c' 'c/*.h' 'c/*.cu' 'c/*.mm' ':!c/tests/*' \
  | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u | wc -l
```

**Какой движок это читает, имеет значение.** Существует четыре двоичных файла (`colibri`,
`kimi_k3`, `inkling`, `olmoe`), и у них НЕТ общего набора ручек — `K3_*`
kimi_k3-only, `INK_*` только намек, и переменная установлена для неправильного движка
молча игнорируется. Запишите владельца при добавлении строки:

```bash
# who reads $V ?
V=K3_BITS; git grep -lE "getenv(_utf8)?\\(\"$V\"" upstream/dev -- 'c/*.c' 'c/*.h' 'c/*.cu' 'c/*.mm' ':!c/tests/*'
```

<a id="step-2-diff-against-whats-documented"></a>
## Шаг 2 — сравнение с документированным

```bash
# vars currently in the code:
git grep -hoE '(compat_)?getenv(_utf8)?\("[A-Z0-9_]+"' upstream/dev -- 'c/*.c' 'c/*.h' 'c/*.cu' 'c/*.mm' ':!c/tests/*' \
  | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u > /tmp/code_vars.txt
# vars currently in the doc (crude: grab `VAR` cells; drop bare numbers, which
# the backtick grep also picks up out of prose):
grep -oE '`[A-Z0-9_]{2,}`' docs/ENVIRONMENT.md | tr -d '`' | grep -vE '^[0-9]+$' | sort -u > /tmp/doc_vars.txt

comm -23 /tmp/code_vars.txt /tmp/doc_vars.txt   # in code, NOT documented -> add these
comm -13 /tmp/code_vars.txt /tmp/doc_vars.txt   # documented, NOT in C -> see below
```

Прежде чем удалять что-либо из второго отчета `comm`, проверьте сторону Python:
`COLI_API_KEY`, `COLI_DEBUG`, `COLI_TOOL_SALVAGE` и друзья настоящие, они есть
просто прочитайте `openai_server.py`/`coli`, а не движком C:

```bash
grep -rn "$V" c/coli c/openai_server.py
```

Он также сообщает о таких вещах, как `O_DIRECT` и `F_NOCACHE`, которые на языке прозы
doc, а не переменные. На самом деле должно идти только имя, которое не читает ни одна программа.

<a id="step-3-update-the-tables-ai-assisted"></a>
## Шаг 3 — обновить таблицы (с помощью ИИ)

Вставьте выходные данные шага 1 в Клод с промптм типа:

> Вот все сайты `getenv()` в `c/glm.c` при фиксации `<hash>` и текущий `ENVIRONMENT.md`.
> Для каждой переменной: подтвердите **по умолчанию** и **эффект** из кода (троичное значение по умолчанию и `/* comment */`).
> Обновите таблицы на месте — сохраните существующую группировку (Общие/Производительность/CUDA/Расширенные/Настраиваемые через CLI), добавьте все новые переменные в нужную группу, удалите те, которые больше не существуют, и исправьте все измененные значения по умолчанию.
> **Не** придумывайте поведение: если комментарий тонкий, описывайте только то, что буквально делает код. Обновите хэш «Сгенерировано из» на `<hash>`.

Тот же шаблон для `SETTINGS.md` с выходом `add_argument`.

Правила, которые помогут сохранить честность:
- **Значения по умолчанию берутся из троичной системы**, например. `getenv("CTX")?atoi(...):4096` → по умолчанию `4096`.
- **Эффект возникает только из кода + встроенный комментарий** — никогда из памяти или предположений.
- Переменная без флага остается только env; обратите внимание на это как таковое.
- Держите ручки эксперимента/отладки в разделе **Дополнительно**, чтобы никто не перепутал их с поддерживаемой поверхностью.

<a id="step-4-verify"></a>
## Шаг 4 — проверка

- Повторно запустите дифф шага 2 — оба выхода `comm` должны быть пустыми.
- Проверьте на глаз 3–5 значений по умолчанию относительно `git grep`.
- Увеличьте строку «Сгенерировано из `main @ <hash>`» в обоих документах.

<a id="cadence"></a>
## Каденс

Обновляйте, когда: релиз удален, или `git grep -c 'getenv(' c/glm.c` изменено, или кто-то сообщает о ручке, которая не задокументирована. Однострочная проверка CI (сравнение количества кодовых переменных с зафиксированным числом) может автоматически отмечать отклонения.