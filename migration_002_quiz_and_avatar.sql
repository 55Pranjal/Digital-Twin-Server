-- ═══════════════════════════════════════════════════════════
-- Migration 002 — quiz_attempts table + avatar_id column
-- Run once in the Supabase SQL editor against the existing project
-- (schema.sql already reflects this for fresh setups).
-- ═══════════════════════════════════════════════════════════

alter table public.students
  add column if not exists avatar_id text not null default 'rogue';

create table if not exists public.quiz_attempts (
  id          bigint generated always as identity primary key,
  student_id  bigint not null references public.students(id) on delete cascade,
  topic       text not null,
  question    text not null,
  answer      text not null,
  correctness double precision not null,
  feedback    text,
  created_at  timestamptz not null default now()
);

create index if not exists quiz_attempts_student_id_created_at_idx
  on public.quiz_attempts (student_id, created_at desc);

alter table public.quiz_attempts enable row level security;
