# ClashAI episode format — `clashai-episode/1`

A whole match as a stream. One JSON object per line:

```
line 1     header
lines 2..n one tick each: a full clashai-observation/1 record + an `episode` block
last line  trailer -- what this recording is actually worth
```

Produce one from the panel: **Vision AI → 6. Record a match for the simulator**.
Leave the video field empty to record the running game; put a path in it to convert a
recorded session instead.

The CLI equivalent, if you want it: `run.py episode` (`--video` / `--interval` / `--minutes`).

---

## Why a stream and not just frames

Four things a simulator needs are **not on any single frame**, by construction:

| | why one frame cannot have it |
|---|---|
| **what was played** | the action is never drawn. It has to be inferred from what CHANGED |
| **velocity** | needs two frames *and* a stable identity |
| **the opponent's elixir** | their bar is not drawn at all — it can only be integrated |
| **who is the same unit as before** | identity is a property of a sequence |

Everything else is already in the per-frame record and is documented in
[OBSERVATION_FORMAT.md](OBSERVATION_FORMAT.md).

---

## The `episode` block

```json
{"schema": "clashai-episode/1", "tick": 41, "t": 14.2, "dt": 0.34,
 "match": {...}, "crowns": {...}, "velocity": {...},
 "deploys": [...], "deploys_suppressed": {...}, "opponent_elixir": {...}}
```

`t` is seconds since the first tick. `dt` is the gap to the previous one and is **not**
constant — the reader takes as long as it takes. Anything per-second must divide by `dt`,
never by a nominal rate.

### `match` — where in the match we are

| field | meaning |
|---|---|
| `seconds_left` | from the on-screen timer panel, or `null` |
| `phase` | `regular` / `overtime`, from the panel's **background colour** |
| `clock_text` | e.g. `"2:37"` — the raw read, for checking us |
| `watched_s` | how long *we* have been recording. Not the same thing |

The screen is preferred over our own elapsed time because it is **absolute**: it stays right
when the recording was trimmed, when we joined late, or when the panel dropped frames.

**Measured.** On two recorded sessions with known frame timestamps, `seconds_left` falls at
**−1.000 / −0.999 seconds per second** (true: −1.000), max residual 0.5 s, **0 outliers in
574 reads**. A misread digit cannot hide from that test — the true value is on a straight
line, and a wrong one is not.

Read rate is **55.9 %** of 401 own-client arena frames and ~81 % of the ticks in a session
recording. It refuses more often than it reads, and that is the intended trade: of the reads,
**0** violate the phase ceiling (regular ≤ 3:00, overtime ≤ 2:00). It also produced **0**
readings across 674 home-screen frames, where a gold counter sits in the same corner in the
same white text — the colon between the minute and the seconds is what rejects those.

### `elixir.multiplier` (in the record, not the episode block)

> **Overtime's FIRST minute is still DOUBLE elixir, not triple.**

| phase | time left | multiplier |
|---|---|---|
| regular | > 1:00 | 1 |
| regular | ≤ 1:00 | 2 |
| overtime | > 1:00 | **2** |
| overtime | ≤ 1:00 | 3 |

When the clock is unreadable the multiplier is `null`, **never 1**. A consumer integrating a
multiplier that is silently too low drifts for the rest of the match with nothing to show for
it. One helper (`observation_export.elixir_multiplier`) serves every path, so the exporter and
the live bot cannot disagree.

### `crowns`

```json
{"mine": 1, "enemy": 0, "towers_unread": 0}
```

No reader of its own: in Clash Royale a crown *is* a felled tower, and the `towers` block
already says which are gone. **`mine` counts crowns WE hold, i.e. ENEMY towers down.**
`towers_unread` is what separates "no crowns yet" from "we could not read two of the towers".

### `velocity`

Per-unit tiles/second, matched by **track id**. The `motion` block in the per-frame record
does the same thing by nearest same-class neighbour and says so; that one swaps two knights
the moment their paths cross, this one cannot, because the same id *is* the same unit. Both
are kept so a consumer without ids still has something.

### `deploys` — what each side played

Two independent signals, reported as different things because they are not equally good:

| `confidence` | source | how much to trust it |
|---|---|---|
| `high` | a hand slot changed from one named card to another | the game telling us we spent something |
| `inferred` | a track id never seen before, and this card was not on the board recently | a real inference |
| `weak` | as above, but the card *was* on the board within the last 10 s | probably the detector re-finding a unit |

Fields: `side`, `card`, `cost`, `tile` (or `xy` when the live bot passes the tap it sent),
`count` and `ids` for a swarm, `last_seen_same_card_s`, `spawn_suspect`, and `evidence` in
words.

**A new track id is not a new unit.** The detector loses units constantly — behind a tower,
inside a push — and once a track ages past `forget_s` the same unit returns with a fresh id.
Before this test, one Archer Queen was reported as **nine separate deploys** in a 100-second
match. The gate is the unit's own top speed over the time it was missing: anything it could
have walked to is a return, anything further is genuinely new. Rejections are counted in
`deploys_suppressed`, never dropped silently.

`spawn_suspect` names a nearby spawner (Tombstone, Witch, Furnace…) when the new unit is
probably that building's free output rather than a card someone paid for. Those are excluded
from the opponent-elixir subtraction but still listed.

**Fifteen skeletons are one deploy.** Units appearing in the same tick within 2.2 tiles are
clustered into a single event with `count`.

### `opponent_elixir`

```json
{"estimate": 4.6, "lower": 4.6, "upper": 10.0, "spent_total": 17.0,
 "unpriced_plays": 1, "rate_per_s": 0.357, "reliability": "estimate"}
```

Starts at 5, regenerates at the base rate times the current multiplier, minus the cost of
every enemy deploy we saw.

**The error only ever goes one way.** Every deploy we miss makes us think they have *more*
than they do. `upper` is the full bar because "they simply did not spend" can never be ruled
out. This is an estimate and says so in `reliability`; it is not a measurement and must not be
trained on as if it were.

---

## The trailer — read this before using a file

```json
{"kind": "trailer", "ticks": 298, "units_per_tick": 0.44,
 "clock_read_rate": 0.809, "hand_read_rate": 0.257,
 "deploys": {"enemy": 18, "mine": 3}, "by_confidence": {...},
 "deploys_suppressed": 10, "warning": "..."}
```

A stream can be perfectly well-formed and still nearly useless. A detector that saw a unit on
one tick in three cannot support an action trace, and you must be able to see that without
re-deriving it from 300 records.

**`warning` is set when `units_per_tick < 1.0`**, and it says what the file is still good for
(a state trace) and what it is not (an action trace).

---

## Where episodes should come from

> **Record LIVE. Converting an old session video gives a much thinner trace.**

Measured with the same detector at the same confidence gate:

| source | units found per frame |
|---|---|
| own-client frames at the size the detector was trained on (669×1182) | **3.56** |
| saved session videos (756×1334) | **0.44** |

The saved recordings are a different resolution and framing from the frames the detector was
trained on, and it is far weaker on them. That is a property of the recordings, not of this
exporter — the same gap shows up when the detector is called directly, with none of this code
in the way. Session videos are still fine as a *state* trace; they are not a basis for action
extraction.

---

## What is still missing

Unchanged from the per-frame format, and none of it is fixed by streaming:

| missing | why |
|---|---|
| **absolute troop HP** | `bars` gives a fraction; multiplying by max HP needs the unit's LEVEL |
| **unit levels** | the only dataset labelling the level badge has 86 frames |
| **status effects** | of seven flags, three have zero positive examples |
| **tower ground position** | only the HP bar's position is known — see the per-frame doc |
| **which card a spell was** | a spell that hits nothing leaves no unit to detect |

One more that is specific to streams: **our own deploys are only `high` confidence when the
hand can be read.** The hand is template matching against `templates/cards/`, so it only names
cards in the configured deck. On a recording of a different deck it reads ~26 % of slots and
our plays fall back to the same inference the enemy's use.
