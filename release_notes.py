"""Build release notes from conventional-commit subjects.

`gh release create --generate-notes` enumerates merged **pull requests** and
nothing else. This repo pushes straight to `main` and has never opened one, so
that path returns an empty "What's Changed" and a bare compare link no matter
how much landed. The changelog lives in the commit subjects instead, so read
those directly.

Run by the release job in `.github/workflows/ci.yml`; stdlib only, so the job
needs no Python setup beyond the runner image's `python3`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import NamedTuple

# Conventional-commit type -> heading, in the order sections are emitted.
# Types deliberately share headings (ci/build); order here is the output order,
# so the entries a reader cares about most come first.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("ci", "CI & Build"),
    ("build", "CI & Build"),
    ("chore", "Chores"),
)

# Anything whose type is absent or unrecognised lands here rather than being
# dropped -- a silently omitted commit is worse than an uncategorised one.
OTHER = "Other Changes"
BREAKING = "Breaking Changes"

# type(scope)!: description
#
# The `(?:\+[a-z]+)*` arm accepts the compound types this repo has actually
# used (`docs+test:`), attributing them to the first type listed.
_SUBJECT = re.compile(
    r"^(?P<type>[a-z]+)(?:\+[a-z]+)*(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<description>.+)$"
)

# A breaking-change footer, per the Conventional Commits spec: the token at the
# start of a line. A bare substring test also matched prose that merely names
# the trailer -- 7757b45's body explains "`BREAKING CHANGE:` trailers promote
# ..." and was listed as a breaking change itself.
_BREAKING_FOOTER = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)

# Record/field separators chosen because git will never emit them itself, so a
# commit message containing newlines (every body) cannot corrupt the split.
_RECORD = "\x1e"
_FIELD = "\x1f"


class Commit(NamedTuple):
    """One parsed commit: `sha` is short, `section` is its output heading."""

    sha: str
    section: str
    description: str
    breaking: bool


def parse_commit(sha: str, subject: str, body: str = "") -> Commit:
    """Classify one commit by its conventional-commit subject.

    A non-conforming subject is kept verbatim under `Other Changes`. Breaking
    changes are flagged by a `!` before the colon or a `BREAKING CHANGE:`
    trailer in the body, and are listed in their own section *as well as* their
    type's, so they cannot be missed by someone skimming headings.
    """
    headings = dict(SECTIONS)
    subject = subject.strip()
    match = _SUBJECT.match(subject)
    breaking = _BREAKING_FOOTER.search(body) is not None

    if match is None:
        return Commit(sha, OTHER, subject, breaking)

    breaking = breaking or match.group("bang") == "!"
    section = headings.get(match.group("type"))
    if section is None:
        # A recognised *shape* but an unrecognised type. Keep the subject
        # verbatim rather than stripping a prefix we do not understand: the raw
        # text is the only signal that the taxonomy needs extending, and an
        # `Other Changes` entry that reads differently from the curated sections
        # is the point.
        return Commit(sha, OTHER, subject, breaking)

    description = match.group("description").strip()
    # Sentence-case the description: subjects are written lowercase-first by
    # convention, which reads badly as a bullet.
    return Commit(sha, section, description[:1].upper() + description[1:], breaking)


def _section_order() -> list[str]:
    """Output order for headings, de-duplicated, with the tail sections last."""
    order: list[str] = [BREAKING]
    for _, heading in SECTIONS:
        if heading not in order:
            order.append(heading)
    order.append(OTHER)
    return order


def render(commits: list[Commit], tag: str, previous: str = "", repo: str = "") -> str:
    """Render grouped commits as the release body.

    An empty `previous` means no prior release, so the trailing link points at
    the tag's full commit list rather than a comparison against nothing.
    """
    grouped: dict[str, list[Commit]] = {}
    for commit in commits:
        grouped.setdefault(commit.section, []).append(commit)
        if commit.breaking:
            grouped.setdefault(BREAKING, []).insert(0, commit)

    lines: list[str] = []
    for heading in _section_order():
        entries = grouped.get(heading)
        if not entries:
            continue
        lines.append(f"### {heading}")
        lines.append("")
        for commit in entries:
            # GitHub auto-links a bare short SHA to its commit within the repo.
            lines.append(f"* {commit.description} ({commit.sha})")
        lines.append("")

    if repo:
        base = f"https://github.com/{repo}"
        link = (
            f"{base}/compare/{previous}...{tag}"
            if previous
            else f"{base}/commits/{tag}"
        )
        lines.append(f"**Full Changelog**: {link}")

    return "\n".join(lines).strip() + "\n"


def read_commits(tag: str, previous: str = "") -> list[Commit]:
    """Read `previous..tag` from git, newest first, merges excluded.

    Requires full history in the checkout -- the release job sets
    `fetch-depth: 0` for exactly this reason.
    """
    revision = f"{previous}..{tag}" if previous else tag
    result = subprocess.run(
        [
            "git",
            "log",
            revision,
            "--no-merges",
            f"--pretty=format:%h{_FIELD}%s{_FIELD}%b{_RECORD}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    commits: list[Commit] = []
    for record in result.stdout.split(_RECORD):
        if not record.strip():
            continue
        sha, subject, body = (record.strip().split(_FIELD) + ["", ""])[:3]
        commits.append(parse_commit(sha, subject, body))
    return commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="tag being released, e.g. v1.2.3")
    parser.add_argument(
        "--previous",
        default="",
        help="previous release tag; omit for the first release",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GH_REPO", os.environ.get("GITHUB_REPOSITORY", "")),
        help="owner/name, for the Full Changelog link",
    )
    args = parser.parse_args(argv)

    sys.stdout.write(
        render(
            read_commits(args.tag, args.previous), args.tag, args.previous, args.repo
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
