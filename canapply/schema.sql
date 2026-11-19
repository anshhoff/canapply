create extension if not exists vector;

-- One row per company job board we watch. Doubles as the source health table.
create table if not exists boards (
  source        text not null,              -- greenhouse | ashby | lever
  token         text not null,
  company       text,
  enabled       boolean not null default true,
  last_run_at   timestamptz,
  last_ok_at    timestamptz,
  last_status   text,                        -- ok | http_404 | error:...
  last_job_count int,
  last_added    int,
  last_closed   int,
  fail_streak   int not null default 0,
  primary key (source, token)
);

create table if not exists jobs (
  id            bigserial primary key,
  source        text not null,
  board_token   text not null,
  ext_id        text not null,
  company       text not null,
  title         text not null,
  url           text not null,
  location_raw  text,
  workplace_raw text,                        -- what the board says: Remote / Hybrid / OnSite
  department    text,
  employment_raw text,
  description   text not null,
  content_hash  text not null,               -- sha1 of title+location+description
  posted_at     timestamptz,
  first_seen_at timestamptz not null default now(),
  last_seen_at  timestamptz not null default now(),
  closed_at     timestamptz,
  -- cheap, rule-based fields filled at ingest time
  is_tech       boolean not null default false,
  level_rule    text,                        -- intern | new_grad | junior | mid | senior | unknown
  countries     text[],                      -- ISO-2 codes parsed from the location string
  -- search
  embedding     halfvec(384),
  embedded_hash text,
  tsv tsvector generated always as (
    setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(company,'')), 'B') ||
    setweight(to_tsvector('english', coalesce(location_raw,'')), 'C') ||
    setweight(to_tsvector('english', left(coalesce(description,''), 20000)), 'D')
  ) stored,
  unique (source, board_token, ext_id)
);
create index if not exists jobs_tsv_idx on jobs using gin (tsv);
create index if not exists jobs_open_idx on jobs (closed_at) where closed_at is null;
create index if not exists jobs_vec_idx on jobs using hnsw (embedding halfvec_cosine_ops);

-- Hard requirements pulled out of a posting. Keyed by content_hash so an unchanged posting is never re-read.
create table if not exists requirements (
  job_id        bigint primary key references jobs(id) on delete cascade,
  content_hash  text not null,
  method        text not null,               -- rules | llm
  model         text,
  data          jsonb not null,              -- see canapply/requirements.py
  quotes_total  int not null default 0,
  quotes_ok     int not null default 0,      -- quotes found verbatim in the posting
  extracted_at  timestamptz not null default now()
);

create table if not exists users (
  chat_id       bigint primary key,
  name          text,
  profile       jsonb not null default '{}', -- country, yoe, grad date, work auth, wants...
  resume_text   text,
  resume_embedding halfvec(384),
  prefs         jsonb not null default '{}', -- things the agent was told to remember
  digest_on     boolean not null default true,
  state         jsonb not null default '{}', -- small conversation state (onboarding step etc.)
  created_at    timestamptz not null default now(),
  last_seen_at  timestamptz not null default now()
);

create table if not exists messages (
  id bigserial primary key,
  chat_id bigint not null,
  role text not null,                        -- user | model | tool
  content jsonb not null,
  at timestamptz not null default now()
);
create index if not exists messages_chat_idx on messages (chat_id, id);

create table if not exists feedback (
  chat_id bigint not null,
  job_id  bigint not null references jobs(id) on delete cascade,
  action  text not null,                     -- up | down | applied
  at timestamptz not null default now(),
  primary key (chat_id, job_id, action)
);

create table if not exists events (
  id bigserial primary key,
  chat_id bigint,
  kind text not null,                        -- search | digest_sent | button | agent_turn | ...
  payload jsonb not null default '{}',
  at timestamptz not null default now()
);

create table if not exists sent_jobs (
  chat_id bigint not null,
  job_id bigint not null references jobs(id) on delete cascade,
  at timestamptz not null default now(),
  primary key (chat_id, job_id)
);

create table if not exists llm_usage (
  id bigserial primary key,
  purpose text not null,
  model text not null,
  prompt_tokens int, output_tokens int,
  ms int,
  ok boolean not null default true,
  at timestamptz not null default now()
);

create table if not exists labels (
  job_id bigint not null references jobs(id) on delete cascade,
  labeler text not null,                     -- 'ansh' | 'judge:<model>'
  bucket text not null,                      -- can_apply | stretch | cannot
  reason text,
  at timestamptz not null default now(),
  primary key (job_id, labeler)
);

-- Postings queued for human labelling (the eval set), in a fixed random order.
create table if not exists eval_pool (
  job_id bigint primary key references jobs(id) on delete cascade,
  ord int not null,
  why text                                   -- how it got into the pool
);

-- Graded relevance of a posting to an eval query (0 = off-topic, 1 = related, 2 = what was asked for).
create table if not exists rel_labels (
  query text not null,
  job_id bigint not null references jobs(id) on delete cascade,
  labeler text not null,
  grade int not null,
  primary key (query, job_id, labeler)
);
