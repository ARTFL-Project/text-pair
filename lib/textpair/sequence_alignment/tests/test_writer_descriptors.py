#!/usr/bin/env python3
"""The chunk writer must not run out of file descriptors and empty whole documents.

    test_writer_descriptors.py

Each document a writer keeps mapped holds a descriptor. At a soft limit of 1,024 the
writer used to fail every document past the thousandth and cache it as unreadable, so
most of eebo_ecco's passages came out as "". Each case runs in a child process under a
descriptor limit small enough to hit, since a limit can be lowered but not raised back.
"""
import multiprocessing as mp
import os
import resource
import sys
import tempfile

from textpair.sequence_alignment.aligner import output

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def corpus(root, count):
    """`count` small documents, each starting with its own number."""
    metas = []
    for number in range(count):
        path = os.path.join(root, f"{number}.txt")
        with open(path, "wb") as handle:
            handle.write(f"{number:06d} some text of document {number}".encode())
        metas.append({"filename": path})
    return [str(number) for number in range(count)], metas


def read_all(root, count, limit, leave=None):
    """In the child: lower both limits to `limit`, and with `leave`, hold descriptors
    open until only that many are free. Maps every document twice over. Returns
    (documents read wrongly, budget, evictions)."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    docs, metas = corpus(root, count)
    holding = []
    if leave is not None:
        held = limit - len(os.listdir("/dev/fd")) - leave
        holding = [open(os.devnull, "rb") for _ in range(held)]
    writer = output.ChunkWriter(docs, metas, root, 300)
    evictions = []
    evict = writer._evict
    writer._evict = lambda keep: (evictions.append(1), evict(keep))
    wrong = 0
    for _ in range(2):
        for slot in range(count):
            if bytes(writer._text(slot)[:6]) != f"{slot:06d}".encode():
                wrong += 1
    for handle in holding:
        handle.close()
    return wrong, writer._budget, len(evictions)


def unreadable(root):
    """In the child: a missing file, an empty one and no filename at all."""
    open(os.path.join(root, "empty.txt"), "wb").close()
    docs = ["missing", "empty", "nameless", "fine"]
    metas = [{"filename": os.path.join(root, "missing.txt")},
             {"filename": os.path.join(root, "empty.txt")},
             {},
             {"filename": os.path.join(root, "fine.txt")}]
    with open(metas[3]["filename"], "wb") as handle:
        handle.write(b"fine")
    writer = output.ChunkWriter(docs, metas, root, 300)
    first = [bytes(writer._text(slot)) for slot in range(4)]
    again = [bytes(writer._text(slot)) for slot in range(4)]
    return first, again, sorted(writer._unreadable)


def raised(hard):
    """In the child: a soft limit of 256 under `hard`. Returns (budget, soft limit)."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    return output._mapping_budget(), resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def in_child(function, *args):
    with mp.get_context("fork").Pool(1) as pool:
        return pool.apply(function, args)


def main():
    with tempfile.TemporaryDirectory() as root:
        # The budget alone keeps the writer inside a small limit: 16 mappings under 80.
        wrong, budget, _ = in_child(read_all, root, 300, 80)
        check(f"a limit of 80 caps the writer at {budget} mappings", budget == 16)
        check("under it, every document reads, twice over", wrong == 0)

        # Something else holding descriptors, so that only 8 of the 16 fit: opening
        # fails inside the budget, and the writer has to evict and retry rather than
        # give the document up.
        wrong, _, evictions = in_child(read_all, root, 300, 80, 8)
        check(f"with 8 descriptors free, the writer evicted to make room ({evictions}x)",
              evictions > 0)
        check("and every document still reads", wrong == 0)

        # A soft limit below the cache with room above it is raised, not lived with.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY or hard >= output._MMAP_CACHE_LIMIT + 64:
            budget, now = in_child(raised, hard)
            check("a low soft limit is raised toward the hard one",
                  budget == output._MMAP_CACHE_LIMIT and now > 256)

        first, again, given_up = in_child(unreadable, root)
        check("missing, empty and nameless documents read as empty",
              first[:3] == [b"", b"", b""])
        check("a readable one beside them reads", first[3] == b"fine")
        check("and the unreadable ones are not retried", again == first and given_up == [0, 1, 2])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
