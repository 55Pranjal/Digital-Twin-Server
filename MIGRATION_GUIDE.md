# EduTwin — Supabase migration guide

Manual steps required to bring the new Supabase-backed auth + database online.
Everything else (schema, server code, client code) is already written — this
is just the account/config setup that has to happen in your own Supabase and
hosting dashboards.

## 1. Create the Supabase project

1. Go to [supabase.com](https://supabase.com) → New project.
2. Once it's provisioned, open **SQL Editor** → New query, paste the contents
   of `schema.sql` (in this directory), and run it. This creates `students`,
   `student_topics`, `student_memory`, and `user_profiles`.
3. Go to **Authentication → Providers** and confirm Email is enabled.
4. Go to **Authentication → Settings**. If you want sign-up to work
   immediately without an email-confirmation step (simplest for a class
   project/demo), turn **Confirm email** off. The app handles either setting,
   but confirmation-required adds an extra "check your email" step.

## 2. Collect your credentials

From **Project Settings → API**:
- `Project URL` → `SUPABASE_URL` (server) / `VITE_SUPABASE_URL` (client)
- `anon public` key → `VITE_SUPABASE_ANON_KEY` (client only)
- `service_role` key → `SUPABASE_SERVICE_ROLE_KEY` (server only — **never** ship this to the client)
- `JWT Settings → Legacy JWT Secret` → `SUPABASE_JWT_SECRET` (server only)

## 3. Configure the server (`Digital Twin(Server side)/.env`)

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=<unchanged>
ALLOWED_ORIGINS=<unchanged>
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...
TEACHER_INVITE_CODE=<pick anything, e.g. a random string — this is what gates teacher sign-up>
```

Install the new dependencies:
```
pip install -r requirements.txt
```

## 4. Configure the client (`Digital Twin(Client side)/.env`)

```
VITE_API_BASE_URL=http://localhost:8000   # or your Render URL
VITE_SUPABASE_URL=...
VITE_SUPABASE_ANON_KEY=...
```

Install the new dependency:
```
npm install
```

## 5. Migrate the existing data

The current `enhanced_students_dataset.csv` and `student_memory.json` are
still in the server directory. With the server's `.env` pointed at your new
Supabase project, run:

```
python migrate_to_supabase.py
```

This copies all students, their topic scores, and their profile-snapshot
history into the new tables, then prints one SQL statement to run in the
Supabase SQL editor (resets the id sequence so new sign-ups don't collide
with migrated student ids).

## 6. Deploy

- **Render (backend)**: add `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
  `SUPABASE_JWT_SECRET`, and `TEACHER_INVITE_CODE` as environment variables
  in the Render dashboard, alongside the existing `GEMINI_API_KEY` and
  `ALLOWED_ORIGINS`.
- **Netlify (frontend)**: add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`
  as build environment variables, alongside the existing `VITE_API_BASE_URL`.

## 7. Try it

1. Visit the site → "I'm a Teacher" → Sign Up tab → enter the
   `TEACHER_INVITE_CODE` you set → you land on the teacher dashboard.
2. Visit "I'm a Student" → Sign Up → then either pick an existing name from
   the migrated roster (claims that student record) or fill in "I'm New
   Here" to create a brand-new one.
3. First page load after the Render backend has been idle will show the
   "Waking up the server…" screen for up to ~60s — that's the free-tier
   cold start, not a bug.
