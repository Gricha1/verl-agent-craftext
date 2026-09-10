# Handoff: verl-agent-caged-craftext / debug_square PPO (для следующего AI-агента)

**Дата документа:** 2026-09-10  
**Автор контекста:** Cursor agent + Gregory Gorbov  
**Чат/транскрипт:** [88c89002 debug_square PPO](88c89002-5500-4fc2-b175-daf69a16b6bc)  
**Workspace:** `/home/gorbov_gv/safe_rl_nlp`  
**Ветка (локально на момент handoff):** `safe`, `HEAD ≈ e1f6a20` (июль 2026). `git fetch` часто **не проходит** без credentials (private GitHub → 404). Working tree обычно **грязный** (много untracked/modified).  
**Comet project:** [gregory-gorbov/verl-agent-caged-craftext](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext)

---

## 0. Как читать этот документ

Это не «идеальная теория проекта», а **фактический handoff** того, что делали в сессии Aug–Sep 2026 на машине **aicenteritl** (Docker `safe_llm_`, conda `verl-agent-311`, GPU A100).

Приоритет фактов:

1. Локальные логи в `logdir/` + параметры в Comet  
2. Текущий код в workspace (особенно `agent_system/reward_manager/episode.py` и yaml dual-конфиги)  
3. Исходный research brief пользователя (ниже) — долгосрочная цель; **практическая работа сессии ушла в credit assignment / dual PPO / 8×8→16×16**

---

## 1. Цели проекта (research)

### 1.1 Главная идея

Проверить, что **одна и та же LLM** может быть одновременно:

- **policy actor**;
- **implicit task-oriented world model** (не Dreamer-reconstruction).

Модель должна предсказывать task-relevant quantities:

| Target | Смысл |
|--------|--------|
| `V_env(s_t)` | env-level value (remaining return) |
| `Q_env(s_t, a_t)` | one-step Q |
| plan-Q / multi-horizon Q | `Q(s_t, a_t…a_{t+k})` |
| done/success | опционально |
| reward prediction | только ablation, не главная линия |

**Не делать:** `state/action → next observation` reconstruction.

Ближе к TD-MPC: `state/action/plan → value, Q, success` (и reward вторично).

### 1.2 Желаемый end-state (статья / диссертация)

Рабочий multi-turn embodied RL pipeline, где:

1. Value/advantage считаются по **env-trajectory**, не по token-response.  
2. Можно честно сравнить:

```text
ordinary PPO (two LLMs: actor + critic)
actor + V_env
actor + V_env + Q_env
actor + V_env + multi-horizon Q_env
```

3. Лестница задач: `debug_square` → sparse collect → craft chain (table / wood pickaxe). Полный Crafter сразу не нужен.

### 1.3 Критическая проблема пайплайна (озвучена в brief)

Каждый env-step = отдельный PPO sample:

```text
prompt = obs_t
response = 1 action token   (response_len=1)
```

Тогда **обычный response-level GAE** даёт `returns ≈ token_score`, а не настоящий `V_env`.

Нужны режимы вроде `env_mc` / `env_gae` (`algorithm.gae_by_trajectory`, token reward `r_t` vs `G_t` vs full episode `R`, actor-value bins и т.д.).

### 1.4 Что уже было до этой сессии (по brief пользователя)

- `debug_square_8x8`, `debug_square_16x16`
- actor-value dual-prompt pipeline
- ordinary dual-LLM PPO baseline
- Q / reward-WM ветки (reward-WM на графиках часто ухудшал обучение)
- GSM8K ablations (LoRA capacity bottleneck и т.п.) — **не главный proof** для env dynamics
- На dense CrafText actor+value иногда лучше ordinary PPO, но это ещё не sparse long-horizon

---

## 2. Инфраструктура aicenteritl (как реально гоняли)

### 2.1 Хост / контейнер

| Что | Значение |
|-----|----------|
| Хост | aicenteritl-подобная машина пользователя (`/home/gorbov_gv/safe_rl_nlp`) |
| Docker | контейнер **`safe_llm_`** |
| Workspace в контейнере | `/usr/home/workspace` (= mount репо) |
| Env | conda **`verl-agent-311`** |
| GPU | обычно 1–2× A100 80GB; **общая карта** — другие юзеры тоже держат процессы (например `lerobot-train`) |
| Rollout engine | **vLLM** |
| Логирование | Comet ML project `verl-agent-caged-craftext` |
| Локальные логи запусков | `logdir/ppo_debug_square_*.log` |
| Чекпоинты PPO dual | **не сохранялись** (`trainer.save_freq=-1`) |

### 2.2 Типичный запуск

```bash
# на хосте / в контейнере
docker exec -it safe_llm_ bash
conda activate verl-agent-311
cd /usr/home/workspace

# 8x8 dual
bash examples/ppo_trainer/ppo_debug_square.sh

# 16x16 dual
bash examples/ppo_trainer/ppo_debug_square_16x16.sh

# 16x16 actor-value (другая линия)
bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh
```

Hyper-yaml:

- `examples/ppo_trainer/config/ppo_debug_square_8x8_dual.yaml`
- `examples/ppo_trainer/config/ppo_debug_square_16x16_dual.yaml`
- `examples/ppo_trainer/config/ppo_debug_square_16x16_actor_value.yaml`

Базовый job (много общих override’ов):  
`examples/ppo_trainer/run_caged_craftext_lora_job.sh`  
там часто жёстко стоит `reward_model.use_episode_return_as_token_reward=False`.

### 2.3 Правила взаимодействия с пользователем (важно!)

Из сессии выучено на практике:

1. **Перед запуском обучения всегда проверять `nvidia-smi`** — если GPU занят чужим процессом, **сразу предупреждать**, не молчать и не «поджимать» память втихую.  
2. Если пользователь сказал **«стоп / не запускай»** — не перезапускать.  
3. Пользователь часто **корректирует интерпретацию Comet/кода** (например, что FINAL dual использовал **per-step `r_t`**, а не full episode return). Считать его domain knowledge приоритетным и перепроверять код+логи.  
4. Нужен **глубокий анализ** (почему critic «мёртвый», почему success не растёт), а не пересказ того, что пользователь уже сказал.  
5. Отвечать **кратко по делу**; длинные отчёты — по запросу / в md / canvas.  
6. **Не коммитить** без явной просьбы.  
7. Веса dual PPO в этих экспериментах **не писались на диск** — только Comet + logdir.

### 2.4 Известные infra-проблемы

- **CUDA OOM** на FSDP↔vLLM `wake_up(kv_cache)` / `cumem_allocator` (особенно 16×16, `gpu_memory_utilization≈0.8`, длинные промпты).  
- NVML/GPU visibility внутри docker иногда ломается — чинили доступом к устройствам.  
- Comet API с хоста часто **timeout / ConnectionError** — ретраи, либо читать `logdir`.  
- GitHub remote private — без auth нельзя подтвердить «latest on GitHub».

---

## 3. Ключевые понятия credit assignment (то, что реально дебажили)

### 3.1 Три разных «token reward»

| Режим | Что кладётся на последний токен ответа | Когда |
|-------|----------------------------------------|--------|
| **`r_t`** | immediate env reward шага | `use_episode_return_as_token_reward=False` в **текущем** `EpisodeRewardManager` |
| **`G_t`** | return-to-go `r_t+…+r_T` | временный локальный путь в коде (логи 1a064d01 / 2a42e153); **сейчас в committed-логике убран** |
| **full `R`** | один и тот же `episode_return` на каждый шаг | `use_episode_return_as_token_reward=True` |

**Важно:** флаг Comet `use_episode_return_as_token_reward=false` **сам по себе не доказывает `r_t`**. В период G_t-экспериментов тот же `False` означал G_t в локальном коде. Смотреть **имя эксперимента / заголовок log** (`token score: per-step r_t` vs `token reward: remaining G_t`).

Текущий код (`agent_system/reward_manager/episode.py`):

```text
False → r_t   (FINAL dual / step-TD style)
True  → full episode return on every step
```

### 3.2 GAE

| Флаг | Эффект при `response_len=1` |
|------|------------------------------|
| `algorithm.gae_by_trajectory=False` | GAE по response; фактически `returns ≈ token_score` |
| `algorithm.gae_by_trajectory=True` | GAE по env-trajectory (`traj_uid`) — «правильный» env-level V, **но** нельзя класть уже готовый `G_t` как step reward (double-count) |

### 3.3 Две архитектуры

| Имя | Смысл |
|-----|--------|
| **dual LLM PPO** | отдельный critic LLM; `use_actor_value_token=False` |
| **actor-value** | shared LLM, value как special token / bins; `use_actor_value_token=True` |

Большая часть **успешных 8×8** в этой линии — **dual + `r_t`**. Actor-value 16×16 в сессии **не взлетел** по success.

### 3.4 Action history

`env.history_length` — сколько уже выполненных действий дописывать в prompt.

- `0` — как исторический FINAL dual  
- `50` — ablation / «память действий»

### 3.5 Среда `debug_square`

- Фиксированная карта, proximity goal, ASCII observation.  
- `max_steps` обычно 50.  
- **16×16:** цель часто **не видна** в локальном окне наблюдения → слепой поиск; в brief предлагались full-map / goal hint / curriculum. Это отдельно от credit-assignment багов.

---

## 4. Хронология работы на aicenteritl (что агент реально делал)

### Phase A — 16×16 actor-value / OOM / GAE double-count (начало сессии)

1. Разбор CUDA OOM в docker/tmux на `ppo_debug_square_16x16_actor_value` (падение на `vllm wake_up(kv_cache)` ~step 78).  
2. Анализ Comet: [3f6f1518](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/3f6f151820dc4ad69b700bff3ddae686) vs [14831ad0](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/14831ad097444a28ab3d2e3d2926aee6).  
3. Выявлено: **`gae_by_trajectory=True` поверх token=`G_t`** → раздутые `critic/returns` (±100…400) при bins `[-26,16]`.  
4. По просьбе: выключили trajectory GAE в 16×16 actor-value конфиге.  
5. Отчёт по 16×16: targets стали ок, но **policy почти не находит цель** (success единицы процентов). Runs вроде [1ac1a602](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/1ac1a602403c44e183f13a9c340040b4).

### Phase B — dual 8×8: history + reward-to-go, потом отладка credit assignment

Пользователь попросил dual PPO 8×8 с:

- history действий в prompt;
- reward-to-go на последнем токене;
- запуск в docker.

Запуски и сравнения с историческими FINAL:

| Key | Имя (Comet) | Конфиг (суть) | Результат |
|-----|-------------|----------------|-----------|
| [1a064d01](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/1a064d01c9ae4e8596a54d06d78f674d) | dual + history + G_t | 8×8, hist=50, G_t, gae_traj=False | плохо учится |
| [2a42e153](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/2a42e153bdb0448789a56140e70fdf50) | то же (повтор/продолжение) | то же | success max ~12%, step~79 |
| [c3a5e727](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/c3a5e727c5414fa3a4e5ff57657351b1) | r_t + traj GAE + history | 8×8, hist=50, r_t, **gae_traj=True** | vf_explained лучше, shaped reward выше, success всё ещё низкий (~≤6%) |
| [983925d3](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/983925d38dda4ef18af90f87919fa37d) | **FINAL dual** (исторический) | 8×8, hist=0, **r_t**, gae_traj=False | **успешный референс** |
| [680dfbf1](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/680dfbf1125a4397bdc8ad3726110d25) | valid-ent вариант FINAL | близко к FINAL | успешный |
| [4be19de5](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/4be19de5e50949359891ce852700d565) | actor-value на 8×8 | actor-value | отдельная линия |

**Важная коррекция пользователя:** у FINAL dual в Comet мог отсутствовать параметр episode-return → агент ошибочно предположил default `True` (полный `R`). Пользователь уточнил: там был **per-step `r_t`**. Это подтверждено воспроизведением.

### Phase C — воспроизведение FINAL и ablation history

| Key | Имя | Конфиг | Результат |
|-----|-----|--------|-----------|
| [02e2744](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/02e2744ddcdb4573ad263487a710e956) | dual + **r_t** + **history0** | 8×8, hist=0, r_t, gae_traj=False | **success → ~99–100%** (~step 100) — воспроизвели FINAL |
| [874bc480](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/874bc480b5224efb9b92795d271ca81c) | dual + r_t + **history** | 8×8, hist=50, r_t | тоже хорошо (~96% к step 90) |

Вывод: на 8×8 рабочий рецепт — **dual + `r_t` + `gae_by_trajectory=False`**; history=50 **не ломает** (в отличие от G_t).

### Phase D — перенос рецепта на 16×16 dual

Создан/обновлён:

- `examples/ppo_trainer/config/ppo_debug_square_16x16_dual.yaml`
- `examples/ppo_trainer/ppo_debug_square_16x16.sh`

| Key | Что случилось |
|-----|----------------|
| [b126b453](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/b126b4532767420f9a0fd031b4d27739) | 16×16 dual r_t+hist50, **2 GPU**, batch 64 → **OOM** ~step 2–4 |
| [da03a539](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/da03a539d25e4c46882ebed96a3cafc4) | рестарт после stop — короткий |
| [da63a962](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/da63a962175b4f1a82606c3fc0fe82d8) | long run: **1 GPU**, batch 32, до **step ~351**, success max **12.5%**, в конце ~0 — **не обучился** |

Инцидент: агент крутил memory-настройки при занятой GPU чужим job’ом → пользователь: **«стоп», «сразу предупреждай», «верни конфиг до OOM»**. Конфиг откатили к `gpu_memory_utilization=0.5`, micro=16, batch=64 (как до OOM-тюнинга); дальнейшие запуски не форсировать без явной просьбы.

### Phase E — housekeeping (сент. 2026)

- Проверка «последний ли git» → fetch без auth ненадёжен.  
- Подтверждение: `da63a962` из этого репо (`logdir/ppo_debug_square_16x16_dual_20260815_105358.log`).  
- Чекпоинты dual PPO: **`save_freq=-1` → весов нет**.

---

## 5. Comet: удобные compare-ссылки

### 5.1 Последняя серия dual (r_t / G_t / 8×8 / 16×16)

[Compare: da63a962, da03a539, b126b453, 874bc480, 02e2744, 2a42e153](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/compare?compareXAxis=step&experiment-tab=panels&experiments=da63a962175b4f1a82606c3fc0fe82d8,da03a539d25e4c46882ebed96a3cafc4,b126b4532767420f9a0fd031b4d27739,874bc480b5224efb9b92795d271ca81c,02e2744ddcdb4573ad263487a710e956,2a42e153bdb0448789a56140e70fdf50&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

### 5.2 G_t / traj-GAE vs FINAL dual

[Compare: 1a064d01, 4be19de5, 983925d3, 680dfbf1](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/compare?compareXAxis=step&experiment-tab=panels&experiments=1a064d01c9ae4e8596a54d06d78f674d,4be19de5e50949359891ce852700d565,983925d38dda4ef18af90f87919fa37d,680dfbf1125a4397bdc8ad3726110d25&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

[Compare: c3a5e727 + FINAL refs](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/compare?compareXAxis=step&experiment-tab=panels&experiments=c3a5e727c5414fa3a4e5ff57657351b1,1a064d01c9ae4e8596a54d06d78f674d,983925d38dda4ef18af90f87919fa37d,680dfbf1125a4397bdc8ad3726110d25,4be19de5e50949359891ce852700d565&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

### 5.3 Ранние 16×16 actor-value

[Compare: 3f6f1518, 14831ad0](https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/compare?compareXAxis=step&experiment-tab=panels&experiments=3f6f151820dc4ad69b700bff3ddae686,14831ad097444a28ab3d2e3d2926aee6&search=rew&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

Связанные: `1ac1a602`, `c997ce45`, `6abd890c` (смотреть по имени `16x16` в проекте).

---

## 6. Таблица экспериментов (сводка)

### 6.1 Dual PPO — локальные логи aicenteritl

| mtime (approx) | Comet key | Log file | Env | hist | token | gae_traj | Итог |
|----------------|-----------|----------|-----|------|-------|----------|------|
| 2026-08-12 | `1a064d01…` | `ppo_debug_square_8x8_dual_20260811_163919.log` | 8×8 | 50 | **G_t** | False | fail |
| 2026-08-13 | `c3a5e727…` | `…_20260811_232728.log` | 8×8 | 50 | r_t | **True** | weak success |
| 2026-08-14 | `2a42e153…` | `…_20260813_010317.log` | 8×8 | 50 | **G_t** | False | fail |
| 2026-08-14 | `02e2744…` | `…_20260813_211140.log` | 8×8 | **0** | **r_t** | False | **OK ~100%** |
| 2026-08-15 | `874bc480…` | `…_20260814_140926.log` | 8×8 | 50 | r_t | False | **OK ~96%** |
| 2026-08-15 | `b126b453…` | `…_16x16_…_092438.log` | 16×16 | 50 | r_t | False | OOM early |
| 2026-08-15 | `da03a539…` | `…_103857.log` | 16×16 | 50 | r_t | False | aborted |
| 2026-08-19 | `da63a962…` | `…_105358.log` | 16×16 | 50 | r_t | False | long, **no learn** |

### 6.2 Исторические / actor-value (важные референсы)

| Key | Роль |
|-----|------|
| `983925d3…` | FINAL dual 8×8 (целевой recipe: r_t, hist≈0) |
| `680dfbf1…` | valid-action entropy dual |
| `4be19de5…` | actor-value 8×8 |
| `14831ad0…` | 16×16 actor-value FINAL-ish; success 0; returns [-1,1]; hist=0 |
| `3f6f1518…` | 16×16 actor-value + traj GAE double-count; OOM |
| `1ac1a602…` | 16×16 actor-value after gae=False; targets OK, policy weak |

---

## 7. Текущее состояние кода / конфигов (ожидаемое)

### 7.1 Рабочий dual recipe (8×8)

```text
dual LLM (separate critic)
reward_model.use_episode_return_as_token_reward=False   # → r_t
algorithm.gae_by_trajectory=False
algorithm.use_actor_value_token=False
env.history_length=0 or 50   # оба ок на 8×8 с r_t
response_len=1, single_token_action
LoRA actor+critic rank 64
trainer.save_freq=-1          # !!! веса не пишутся
```

Файлы:  
`examples/ppo_trainer/config/ppo_debug_square_8x8_dual.yaml`  
`examples/ppo_trainer/ppo_debug_square.sh`

### 7.2 16×16 dual (тот же recipe, не взлетел)

```text
env: debug_square_16x16
history_length=50
max_prompt_length=1536
gpu_memory_utilization=0.5   # откат после OOM/чужой занятости
```

Файлы:  
`examples/ppo_trainer/config/ppo_debug_square_16x16_dual.yaml`  
`examples/ppo_trainer/ppo_debug_square_16x16.sh`

### 7.3 Открытые исследовательские вопросы

1. **Почему 16×16 dual с тем же recipe не учится?**  
   - partial observability / цель вне окна  
   - более длинный horizon / exploration  
   - batch/GPU (1 vs 2) / нестабильность  
   - недостаточно шагов / другой shaped reward scale  

2. **Как получить настоящий `V_env` без ломки обучения?**  
   - `r_t` + `gae_by_trajectory=True` на 8×8 дал лучший vf, но success не взлетел (c3a5e727)  
   - `G_t` + response GAE — плохой MC proxy при hist и без traj GAE  

3. Вернуться к **actor-value + env_mc/env_gae + Q/plan-Q** после стабильного baseline.  

4. Включить **`trainer.save_freq>0`**, иначе успешные прогоны нельзя валидировать/продолжать.

5. Observation fix для 16×16 (full-map / goal direction / curriculum).

---

## 8. Метрики, на которые смотреть в Comet

Обязательные:

- `episode/success_rate`
- `episode/reward/mean` (и min/max)
- `critic/score/mean|min|max` — должен совпадать по смыслу с token reward
- `critic/returns/mean|min|max` — при gae_traj=False ≈ score; при double-count — взрывается
- `critic/values/*`, `critic/vf_explained_var`
- `actor/entropy_loss`, `actor/pg_loss`, `actor/kl_loss`
- `prompt_length/mean` (history удлиняет промпт)
- `training/global_step`, `perf/max_memory_*`

Для actor-value дополнительно:

- `actor/value_token_accuracy`, `actor/value_token_loss`
- попадание targets в bins (`value_return_min/max`)

**Диагностика:** ранний `vf_explained≈0` у FINAL тоже бывает — **сам по себе не диагноз**. Смотреть success takeoff и диапазон returns vs bins.

---

## 9. Как агенту работать дальше (playbook)

### 9.1 Перед любым train launch

```bash
nvidia-smi
docker ps | grep safe_llm
docker exec safe_llm_ bash -lc 'nvidia-smi; ps aux | grep -E "main_ppo|ray|vllm" | grep -v grep | head'
```

Если GPU занят чужим процессом → **сообщить пользователю и ждать**.

### 9.2 Сопоставить Comet run ↔ локальный лог

```bash
grep -l '<experiment_key>' /home/gorbov_gv/safe_rl_nlp/logdir/*.log
head -40 <logfile>   # HISTORY / token score / gae
```

### 9.3 Читать параметры Comet

Использовать `comet_ml.api.API` с `COMET_API_KEY` из окружения (не хардкодить ключи в новые файлы). При timeout — retry / опираться на logdir.

### 9.4 Не путать линии

- Dual `r_t` 8×8 = **рабочий baseline**.  
- Dual `G_t` = провал в этой серии.  
- Actor-value 16×16 = другая задача + observability.  
- Отсутствие параметра в Comet ≠ доказательство default.

### 9.5 Язык общения

Пользователь пишет по-русски, коротко, часто кидает Comet compare URL. Ожидает: вердикт → факты → что делать дальше. Без воды.

---

## 10. Ключевые файлы репозитория

| Path | Зачем |
|------|--------|
| `agent_system/reward_manager/episode.py` | token reward r_t vs episode R |
| `verl/trainer/ppo/core_algos.py` | `compute_gae_advantage_return_by_trajectory` |
| `agent_system/environments/env_manager.py` | prompts, history |
| `examples/ppo_trainer/run_caged_craftext_lora_job.sh` | базовые overrides |
| `examples/ppo_trainer/ppo_debug_square.sh` | dual 8×8 entry |
| `examples/ppo_trainer/ppo_debug_square_16x16.sh` | dual 16×16 entry |
| `examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh` | actor-value 16×16 |
| `examples/ppo_trainer/config/ppo_debug_square_*_dual.yaml` | hyper params |
| `logdir/` | локальные stdout логи запусков aicenteritl |
| `training_checkpoints/` | сейчас в основном reward_wm SFT, **не** dual PPO |

---

## 11. Что НЕ сделано / не оставлять как «готово»

- [ ] Env-level actor-value/Q pipeline как в исходном brief (полноценно)  
- [ ] Stable 16×16 learning (dual или actor-value)  
- [ ] Observation fix для 16×16  
- [ ] Sparse achievement ladder  
- [ ] Сохранение чекпоинтов dual PPO  
- [ ] Чистый sync с GitHub tip  
- [ ] Валидационные видео в Comet — пользователь жаловался, что не логируются  

---

## 12. Короткий TL;DR для следующего агента

> На aicenteritl в Docker `safe_llm_` крутили VERL PPO для CrafText `debug_square`. Долгосрочная цель — shared LLM как actor + task-oriented V/Q. Практически отладили **credit assignment**: рабочий dual baseline на **8×8 = per-step `r_t` + `gae_by_trajectory=False` (+/− history)** — success ~100% (`02e2744`, `874bc480`, референс `983925d3`). **`G_t` и часть traj-GAE постановок на 8×8 не дали takeoff.** Перенос того же dual recipe на **16×16** (`da63a962`) до step 351 **не обучился**; ранние 16×16 actor-value ломались OOM/double-count GAE и partial observability. Веса dual **не сохранялись**. Перед запусками **проверять занятость GPU** и не стартовать без явной просьбы пользователя.

---

## 13. Приложение: имена экспериментов ↔ гипотезы

| Гипотеза | Эксперимент-проверка | Вердикт |
|----------|----------------------|---------|
| FINAL = r_t + hist0 | `02e2744` vs `983925d3` | подтверждено |
| history ломает r_t | `874bc480` | **не ломает** на 8×8 |
| G_t лучше для V | `1a064d01`, `2a42e153` | **хуже** для success |
| traj GAE + r_t = правильный V и success | `c3a5e727` | V лучше, success всё ещё слабый |
| тот же dual recipe масштабируется на 16×16 | `b126b453`→`da63a962` | **пока нет** |
| traj GAE поверх G_t на actor-value 16×16 | `3f6f1518` | ломает returns (double-count) |

---

*Конец handoff. При сомнениях сверяй Comet key с `logdir/` header и текущий `episode.py`, а не только имя флага в UI.*
