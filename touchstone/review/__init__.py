"""Review: the room over finished Harbor tasks and trials (voice or text).

`trials.py` reads the job directories and orders which trial to walk through; `changes.py` turns a
verifier correction into an edit of a task's `tests/` files; `regrade.py` reruns `harbor job
regrade` and diffs the rewards; `agent.py` is the review agent the room drives. Everything the room
decides is a row in `store.reviews` and a file change under the dataset — nothing else.
"""
