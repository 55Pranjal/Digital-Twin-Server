# EduTwin — Changes Changelog

> **Files changed:** `digital_twin.py` · `main.py`  
> **Date:** 2026-04-02  
> **Summary:** Added full Student CRUD layer — create, read (list), update, and delete students — plus the corresponding FastAPI endpoints so the frontend can manage students directly through the REST API.

---

## `digital_twin.py`

### New Section — Student CRUD (`lines 100 – 244`)

A brand-new *Student CRUD* section was inserted between the data-loading helpers and the memory management block.

#### `list_students() → list[dict]`
Returns a lightweight summary of every student in the dataset. Only the five key metadata columns are included (`student_id`, `name`, `year`, `branch`, `archetype`), making it cheap to populate dashboards and dropdown menus without building full per-student profiles.

#### `add_student(student_data: dict) → dict`
Adds a new student to the persisted CSV dataset.

| Accepted key | Type | Default |
|---|---|---|
| `name` | `str` | *(required)* |
| `year` | `int` | `1` |
| `branch` | `str` | `"CSE"` |
| `archetype` | `str` | `"unknown"` |
| `topics` | `list[dict]` | `[]` *(all 8 topics auto-filled with neutral defaults)* |

- Auto-assigns the next available `student_id` (`max existing + 1`).
- Ensures all eight `COURSE_TOPICS` rows are always created, even if the caller supplies fewer.
- Immediately builds the LLP profile, writes the memory entry, and persists both the CSV and `student_memory.json`.
- **Returns** the freshly built profile dict.

#### `delete_student(student_id) → bool`
Permanently removes all rows for the given student from the CSV and clears their entry from the in-memory history / `student_memory.json`.  
Returns `True` on success, `False` if the ID was not found.

#### `update_student(student_id, updates: dict) → dict`
Partially updates a student without touching fields that are not supplied.

| Updatable metadata | Updatable per-topic fields |
|---|---|
| `name`, `year`, `branch`, `archetype` | `score`, `time_spent`, `attempts`, `confidence`, `engagement`, `fatigue` |

- Raises `ValueError` if the student does not exist.
- Invalidates the cached profile so the next build reads fresh CSV data.
- Persists the updated CSV and memory, then **returns** the rebuilt profile.

---

## `main.py`

### New Pydantic Models (`lines 157 – 187`)

Four new request-body models were added to support the student CRUD endpoints:

| Model | Purpose |
|---|---|
| `TopicInput` | Per-topic data when **creating** a student (all fields have defaults) |
| `AddStudentBody` | Full body for `POST /api/v1/students` |
| `TopicUpdate` | Per-topic data when **updating** a student (all fields optional) |
| `UpdateStudentBody` | Full body for `PUT /api/v1/students/{student_id}` |

### New Endpoints

#### `GET /api/v1/students` — Endpoint 13
```
Handler: get_all_students()
```
Returns `{ total, students[] }` — a lightweight list of all students. Designed for frontend dropdowns and overview tables. Logs and wraps any dataset error into a `500` response.

---

#### `POST /api/v1/students` — Endpoint 14  *(status 201)*
```
Handler: create_student(body: AddStudentBody)
```
Creates a new student from the request body.

- A unique `student_id` is assigned server-side (never sent by the client).
- Supplying fewer than 8 topics is fine — the engine fills in the rest with neutral defaults.
- **Returns** `{ message, student_id, profile }`.

**Example request body:**
```json
{
  "name": "Alice",
  "year": 2,
  "branch": "ECE",
  "archetype": "visual_learner",
  "topics": [
    { "topic": "ai_ml", "score": 0.75, "confidence": 0.70 },
    { "topic": "probability", "score": 0.45 }
  ]
}
```

---

#### `PUT /api/v1/students/{student_id}` — Endpoint 15
```
Handler: modify_student(student_id, body: UpdateStudentBody)
```
Partially updates an existing student. Only the keys present in the body are changed.

- `None` fields in the body are stripped before being passed to `update_student`.
- Per-topic updates are also stripped of `None` values so only explicitly supplied fields overwrite the CSV.
- Returns `404` if the student is not found, `500` on unexpected errors.
- **Returns** `{ message, student_id, profile }`.

**Example request body:**
```json
{
  "name": "Alice Updated",
  "topics": [
    { "topic": "ai_ml", "score": 0.82 },
    { "topic": "probability", "confidence": 0.65 }
  ]
}
```

---

#### `DELETE /api/v1/students/{student_id}` — Endpoint 16  *(status 200)*
```
Handler: remove_student(student_id)
```
Permanently deletes a student from the CSV and their memory history.

- Returns `404` if the ID does not exist.
- **Returns** `{ message, student_id }` on success.

---

### Updated Imports in `main.py`

The following names were added to the `from digital_twin import (...)` block to expose the new CRUD layer to the API layer:

```python
add_student
delete_student
list_students
update_student
```

---

## Complete Endpoint Reference (after changes)

| # | Method | Path | Description |
|---|---|---|---|
| 1 | GET | `/api/v1/profile/{student_id}` | Build / fetch Live Learner Profile |
| 2 | GET | `/api/v1/diagnose/{student_id}` | LLM weakness diagnosis |
| 3 | GET | `/api/v1/explain/{student_id}/{concept}` | Personalised concept explanation |
| 4 | GET | `/api/v1/predict/{student_id}/{topic}` | Performance prediction |
| 5 | POST | `/api/v1/exam/{student_id}` | Exam answer simulation |
| 6 | GET | `/api/v1/plan/{student_id}` | Study plan generation |
| 7 | GET | `/api/v1/simulate/{student_id}` | Run all intervention simulations |
| 8 | POST | `/api/v1/simulate/{student_id}/apply` | Apply a chosen intervention |
| 9 | POST | `/api/v1/feedback/{student_id}` | Feedback / confidence correction |
| 10 | GET | `/api/v1/recommend/{student_id}` | Rule-based recommendations |
| 11 | GET | `/api/v1/teacher` | Class-wide teacher overview |
| 12 | POST | `/api/v1/evaluate` | Full evaluation pipeline |
| **13** | **GET** | **`/api/v1/students`** | **List all students** *(new)* |
| **14** | **POST** | **`/api/v1/students`** | **Add a new student** *(new)* |
| **15** | **PUT** | **`/api/v1/students/{student_id}`** | **Update a student** *(new)* |
| **16** | **DELETE** | **`/api/v1/students/{student_id}`** | **Delete a student** *(new)* |
| — | GET | `/health` or `/` | Health check |
