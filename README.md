# CanApply

**A job search that tells you which jobs you can actually get.**

Most job search ranks by how well a posting matches your resume. For a student that is the easy half. The
hard half is that most "matches" are jobs you cannot get: "Remote" that turns out to be US-only, "3+ years"
buried in the fourth paragraph, an internship that needs enrolment in another country.

CanApply reads the hard requirements out of every posting, checks them against your profile in plain code,
and sorts jobs into three buckets:

- ✅ **Can apply**: you meet every stated requirement
- 🟡 **Stretch**: you miss exactly one soft requirement, and it tells you which
- ⛔ **Can't**: with the sentence from the posting that rules you out

It runs as a Telegram bot: [@CanApplyJobsBot](https://t.me/CanApplyJobsBot).

> Status on 19 Nov 2026: built and running locally. Not yet deployed for other people.

## What the data says

For one real profile (a final-year student in India with under a year of experience), over
**16,431 open tech postings** from 349 company career pages:

| Verdict for this person | Postings | Share |
|---|---|---|
| ✅ Can apply | 185 | 1.1% |
| 🟡 Stretch | 1,701 | 10.4% |
| ⛔ Can't | 14,545 | 88.5% |

- Of the **242 remote postings** read in full, **181 (75%)** are closed to someone in India because of where the
  hire must live. "Remote" usually means "remote within one country".
- A plain hybrid search put **0.25** jobs this person can apply to in its top 10. With eligibility in the
  pipeline that became **7.7 to 8.1** (and 3.3 to 3.7 with the relevance floor that is on by default; see Results).

(Only 1,487 of these postings have a full LLM reading so far; the other 14,944 are judged from rules, which were
85% accurate on the labelled sample. So read the first table as a close estimate, not an exact count.)

So plain semantic search, however good, mostly returns jobs this person cannot get. That is the problem
this project is built around.

## How it works

```
company career pages                    ┌──────────── every 6 h (GitHub Actions) ────────────┐
(Greenhouse, Ashby, Lever) ──► ingest ──► rules reading ──► embeddings ──► LLM reading (bounded)
                                 │            │                 │                │
                                 ▼            ▼                 ▼                ▼
                              Postgres:  jobs + full-text index + pgvector + requirements (with quotes)
                                                        │
        Telegram ──► agent (tool loop) ──► search: keyword + vector ─► RRF ─► eligibility ─► re-rank
                          │                                                        │
                          └── remembers preferences, corrects the profile          └── pure code, no LLM
```

1. **Ingest** (`canapply/ingest.py`). Pulls the public job-board APIs of 349 companies. New postings are
   added, changed ones updated, and postings that disappeared are closed. Each source has a health row
   (last success, jobs added and closed, failure streak), exposed at `/status`.
2. **Read the requirements, cheapest way first** (`canapply/extract.py`).
   - *Rules* run on every posting for free: level from the title, "N years" patterns, the location string,
     work-authorisation phrases.
   - *The LLM* reads the postings that matter, with structured output: minimum years, where you must be,
     visa and work-authorisation rules, student and graduation-year rules, degree, must-have skills. Senior-titled
     roles never get an LLM call, because the title already settles them.
   - *Postings nobody has read yet* get their LLM reading lazily, at the moment they are about to be shown.
   - Results are cached by a hash of the posting, so an unchanged posting is never read twice.
3. **Do not trust the model** (`extract.verify`). Every requirement must come with a quote. Code then checks
   that the quote (a) appears verbatim in the posting and (b) actually contains the evidence: the number of
   years, a word like "sponsor", a word like "worldwide". A requirement that fails either check is dropped.
4. **Decide in code** (`canapply/eligibility.py`). The LLM only reads. The verdict comes from about 150 lines
   of plain Python that take a profile and a set of requirements and return a bucket with reasons. It is
   unit-tested and gives the same answer every time.
5. **Search** (`canapply/search.py`). Postgres full-text search and pgvector cosine search run side by side
   and are merged with Reciprocal Rank Fusion. Eligibility is applied *inside* retrieval as a SQL filter, not
   only afterwards, then an LLM re-ranks the survivors against the resume. Thumbs up and down shift the query
   vector toward and away from those jobs.
6. **Agent** (`canapply/agent.py`). A hand-written tool loop with five tools (`search_jobs`, `explain_job`,
   `skills_gap`, `remember`, `update_profile`), a step limit, and saved preferences. It can only talk about
   jobs the tools returned, and it cannot overrule a verdict.

## Results

All numbers below were measured on 18 and 19 Nov 2026 and can be reproduced with `python -m evals.run_evals`.

### 1. Does the reading get the verdict right?

120 postings, sampled in fixed strata (India, remote, intern and new grad, other, senior titles). Gold verdicts
come from an independent labeller reading each full posting against a written rubric
([`evals/RUBRIC.md`](evals/RUBRIC.md)) without seeing the system's answer.

| Reading | Verdict accuracy | "Can apply" precision | "Can apply" recall |
|---|---|---|---|
| Rules only | 83.3% | 5 of 12 | 5 of 7 |
| LLM reading + quote checks | **91.7%** | 6 of 8 | 6 of 7 |
| Rules only, after the threshold fix below | 85.0% | 6 of 13 | 6 of 7 |
| LLM reading + quote checks, after the fix | **94.2%** | 7 of 9 | 7 of 7 |

The first two rows are what the system scored the first time it was measured. The last two are after fixing
a bug the eval itself exposed, so they are measured on the same postings the fix was found on.

Quote checks: in this sample 120 of 123 quoted requirements passed. Across the whole index 1,247 of 1,287
passed (97%), which means 40 claims by the model were thrown out by code before they could affect a verdict.

A second, larger check: across all 490 postings labelled for the two evals, the system's verdict matched the
label on 91.2%, with "can apply" precision 58 of 74 and recall 58 of 62.

What the mistakes were, read one by one:

The first run had 10 disagreements out of 120:

- **4 were my code.** Three postings asked for "2+ years" and one for "1+ years". The candidate has 0.8. My
  thresholds called a 1.2-year gap a hard miss and a 0.2-year gap a soft miss, which contradicted my own rubric.
  Fixed: within half a year passes, under two years short is a stretch.
- **2 were the labeller, not the system.** One internship says "We do sponsor ... employment visas for this
  role" and the labeller still applied the "US internship" rule. One posting was shown to the labeller with its
  middle cut out, so the labeller missed a "3+ years" line that the system quoted.
- **2 were reading misses.** "Graduating by June 2028" was read as 2028 only. A "third-year students or below"
  rule on an internship was not captured at all, because the schema has no field for year of study.
- **2 are judgement calls**: a role offered "in three US offices or remote within the US or Canada", and a role
  with lead duties but no "lead" in the title.

### 2. Of the top 10 results, how many can this person apply to?

12 searches an early-career person would type ([`evals/queries.json`](evals/queries.json)), run through eight
versions of the pipeline. Every posting any version returned was labelled for eligibility, and each
search-posting pair was graded for relevance (0, 1 or 2).

| Pipeline | Can apply, of 10 | Can apply or stretch, of 10 | Relevant **and** can apply, of 10 | Relevant, share of results | NDCG@10 |
|---|---|---|---|---|---|
| Keyword only | 0.0 | 0.4 | 0.0 | 45% | 0.28 |
| Vector only | 0.2 | 2.2 | 0.2 | 94% | 0.80 |
| Hybrid (keyword + vector, RRF) | 0.25 | 1.75 | 0.25 | 90% | 0.77 |
| Hybrid + LLM re-rank | 0.25 | 2.1 | 0.25 | 94% | **0.84** |
| Hybrid + eligibility | 7.7 | 8.6 | 3.4 | 44% | 0.27 |
| Full pipeline (eligibility + lazy reading + re-rank) | **8.1** | **8.75** | **4.25** | 48% | 0.29 |
| Hybrid + eligibility + relevance floor (added after seeing the rows above) | 3.7 | 8.4 | 2.5 | 80% | 0.55 |
| Full pipeline + relevance floor (what the bot runs; measured a day later) | 3.3 | 8.3 | 2.4 | 83% | 0.52 |

How to read it:

- **Better retrieval does not fix the real problem.** Re-ranking lifts NDCG from 0.77 to 0.84, and the person can
  still apply to only a quarter of one job in the top 10. Relevance was already fine. Eligibility was not.
- **Eligibility changes the answer**: from 0.25 to about 8 jobs out of 10 the person can apply to, and from
  0.25 to 4.25 that are both relevant and within reach.
- **It costs relevance, and I measured how much.** For searches with few reachable matches ("ios developer",
  "founding engineer") the eligibility pipeline filled the list with reachable but unrelated jobs: only 44 to
  48% of its results were relevant.
- **The relevance floor trades the other way.** It brings relevant results back to 80%, but it also drops some
  relevant "can apply" jobs, so that count falls from 7.7 to 3.7 and the gap is filled with relevant stretch
  jobs. There are simply few jobs that are both. The floor is on by default in the bot, because showing ten
  unrelated jobs breaks trust faster than showing four good ones.
- **The re-ranker adds little once the floor is on.** With both, 83% of results are relevant (80% without the
  re-ranker) and the "can apply" count is 3.3 against 3.7: inside the noise for 12 searches. So the shipped
  pipeline puts about 2.4 jobs that are both relevant and within reach in the top 10, against 0.25 for plain
  hybrid search, and most of the remaining slots are relevant stretch jobs. A floor tuned per bucket is the
  next thing to try.

### Honest limits of these numbers

- The gold labels were written by Claude (a different model family from the Gemini models under test), not by
  a person. The bot has a `/label` mode for collecting human labels on the same 120 postings; until those exist,
  treat the labels as a strong proxy and not as ground truth. As of 19 Nov 2026 there are no human labels yet.
- One profile. The eligibility code is general (a second test profile, a UK citizen with two years of
  experience, gets UK and Europe roles), but the evals were only scored for one person.
- The 120-posting sample has only 7 true "can apply" postings, so precision and recall on that class move a lot
  with one posting either way.
- Relevance grades were given mostly from the title and the first lines of each posting. A few labellers judged
  senior-titled postings from the title and location without reading the whole text.
- 1,487 of 16,431 postings have a full LLM reading. Every intern, new-grad and junior-titled posting is
  covered; the rest are judged by rules until they are about to be shown to someone.
- The eligibility rules encode judgement (for example "one year short is a stretch"). Someone else could
  reasonably draw those lines differently.

## Design choices I would defend

| Choice | Why |
|---|---|
| The LLM extracts, code decides | A verdict I can unit-test and explain line by line. The model cannot talk itself into "you can apply". |
| Quotes checked twice in code | Catches invented quotes and also real quotes that do not support the claim. The second check was added after the model called a job "remote anywhere" while quoting a location line that listed four regions. |
| Eligibility inside retrieval | Filtering after retrieval failed: when about 90% of the nearest postings are out of reach, the few reachable ones never make the candidate list. |
| Reciprocal Rank Fusion | Full-text and cosine scores are on different scales. Ranks can be combined without tuning weights. |
| Rules first, LLM second, lazy third | The free tier allows 500 requests a day per model. Spending them only where rules cannot decide is what makes 16k postings workable. |
| Local embedding model | The free embedding API allowed about 45 postings a minute. A small local model embedded the whole index in 28 minutes and search no longer depends on an API quota. |
| No agent framework | The loop is 40 lines. I can say exactly when it calls a tool, how it stops, and what it does when a tool fails. |
| Fallback chain per model | Free-tier models answered "overloaded" on about half of calls. Every LLM job has at least two models, and live requests fail over after 18 seconds. |

## Run it

```bash
brew install postgresql@17 pgvector && brew services start postgresql@17
createdb canapply
cp .env.example .env            # add GEMINI_API_KEY and TELEGRAM_BOT_TOKEN
uv sync

uv run python -m canapply.ingest --discover    # pull every board in data/boards.txt
uv run python -m canapply.extract --rules      # free rules reading for everything
uv run python -m canapply.embed                # local embeddings
uv run python -m canapply.extract --limit=500  # LLM reading for the postings that matter most
uv run python -m canapply.bot                  # the Telegram bot

uv run pytest                                  # 39 tests, no database or API key needed
uv run python -m evals.run_evals               # recompute the tables above
```

`ci.yml` runs lint, the unit tests, an eval gate that fails the build if verdict accuracy on the frozen
120-posting sample drops below 92%, and a Docker build that starts the container and checks it answers.

## Hosting it for free

One Docker image runs the bot and a small status API (`/ping`, `/health`, `/status`, `/search`).

- **Database:** any Postgres with pgvector. `scripts/copy_db_to_hosted.sh` copies the local database into an
  empty hosted one (written for Neon's free plan).
- **Host:** `render.yaml` describes one free Render web service. When the host gives the service a public URL,
  the bot switches from long polling to a **Telegram webhook**, so an incoming message wakes a sleeping free
  instance, and it pings its own `/ping` every 10 minutes so it rarely sleeps at all. `/ping` does not touch the
  database, which lets a serverless database stay asleep between messages.
- **Refresh:** `.github/workflows/refresh.yml` re-reads every career page every 6 hours. It stays off until the
  repository variable `REFRESH_ENABLED` is `true` and the `DATABASE_URL` and `GEMINI_API_KEY` secrets exist.

## Layout

```
canapply/
  ingest.py  sources/boards.py   pull + upsert + close postings, per-source health
  rules.py   geo.py              stage 1: free rules (title level, years, location, work authorisation)
  extract.py requirements.py     stage 2: LLM reading with quotes, the code checks, the cascade
  eligibility.py                 profile + requirements -> can apply / stretch / can't, with reasons
  embedder.py embed.py           local embeddings of the role section of each posting
  search.py                      keyword + vector + RRF, eligibility in retrieval, re-rank, personalisation
  agent.py   gap.py              the tool loop; "what should I learn next" from the data
  bot.py     telegram.py         Telegram: onboarding, chat, daily picks, feedback, labelling mode
  api.py     serve.py  llm.py    status API; one-process server; Gemini over plain HTTPS with pacing and fallback
evals/                           rubric, queries, judge, harness, frozen fixture, results
tests/                           rules, eligibility, quote checks, RRF, eval regression gate
```
