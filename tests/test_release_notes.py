"""Tests for release_notes.py, which builds the body of every GitHub Release.

Nothing else exercises this module: it runs only inside the release job, and by
the time a bad body is visible it has already been published. These are the
canary.
"""

from __future__ import annotations

from pathlib import Path

from release_notes import (
    BREAKING,
    OTHER,
    SECTIONS,
    Commit,
    parse_commit,
    render,
)


class TestParseCommit:
    def test_plain_type(self) -> None:
        commit = parse_commit("abc1234", "fix: pin voice loading to our snapshot")
        assert commit.section == "Bug Fixes"
        assert commit.description == "Pin voice loading to our snapshot"
        assert commit.breaking is False

    def test_scope_is_stripped_from_the_description(self) -> None:
        # `fix(docs):` appears in this repo's history; the scope is routing
        # information, not part of the sentence a reader wants to see.
        commit = parse_commit("abc1234", "fix(docs): correct the venv-move step")
        assert commit.section == "Bug Fixes"
        assert commit.description == "Correct the venv-move step"

    def test_compound_type_uses_the_first(self) -> None:
        # `docs+test:` is real history (0c9f41a). It must not fall through to
        # Other Changes just because it names two types.
        commit = parse_commit("abc1234", "docs+test: apply review findings")
        assert commit.section == "Documentation"

    def test_ci_and_build_share_a_heading(self) -> None:
        assert parse_commit("a", "ci: x").section == "CI & Build"
        assert parse_commit("b", "build: y").section == "CI & Build"

    def test_unknown_type_is_kept_verbatim_not_dropped(self) -> None:
        # A silently omitted commit is worse than an uncategorised one: the
        # notes would understate what shipped with no signal that they had.
        commit = parse_commit("abc1234", "wibble: something unconventional")
        assert commit.section == OTHER
        assert commit.description == "wibble: something unconventional"

    def test_non_conventional_subject_is_kept_verbatim(self) -> None:
        commit = parse_commit("abc1234", "Initial commit")
        assert commit.section == OTHER
        assert commit.description == "Initial commit"

    def test_bang_marks_breaking(self) -> None:
        commit = parse_commit("abc1234", "feat!: drop Python 3.11")
        assert commit.breaking is True
        assert commit.section == "Features"
        assert commit.description == "Drop Python 3.11"

    def test_bang_with_scope_marks_breaking(self) -> None:
        assert parse_commit("a", "feat(api)!: rename the flag").breaking is True

    def test_breaking_change_trailer_in_body_marks_breaking(self) -> None:
        commit = parse_commit("a", "feat: rework", "BREAKING CHANGE: config moved")
        assert commit.breaking is True

    def test_description_is_sentence_cased(self) -> None:
        assert parse_commit("a", "docs: add screenshots").description.startswith("Add")

    def test_sentence_casing_leaves_acronyms_alone(self) -> None:
        assert (
            parse_commit("a", "ci: SHA-pin checkout").description == "SHA-pin checkout"
        )


class TestRender:
    def test_sections_follow_declared_order(self) -> None:
        commits = [
            parse_commit("c1", "chore: bump deps"),
            parse_commit("c2", "fix: a bug"),
            parse_commit("c3", "feat: a feature"),
        ]
        body = render(commits, "v2.0.0", "v1.0.0", "o/r")
        assert body.index("### Features") < body.index("### Bug Fixes")
        assert body.index("### Bug Fixes") < body.index("### Chores")

    def test_empty_sections_are_omitted(self) -> None:
        body = render([parse_commit("c1", "fix: a bug")], "v1.0.1", "v1.0.0", "o/r")
        assert "### Bug Fixes" in body
        assert "### Features" not in body
        assert OTHER not in body

    def test_breaking_changes_lead_and_are_also_listed_under_their_type(self) -> None:
        # Listed twice on purpose: someone skimming headings must not be able to
        # miss a breaking change, and someone reading Features expects it there.
        commits = [parse_commit("c1", "feat!: drop Python 3.11")]
        body = render(commits, "v2.0.0", "v1.0.0", "o/r")
        assert body.index(f"### {BREAKING}") < body.index("### Features")
        assert body.count("Drop Python 3.11") == 2

    def test_compare_link_when_a_previous_tag_exists(self) -> None:
        body = render([parse_commit("c1", "fix: x")], "v1.1.0", "v1.0.0", "o/r")
        assert body.rstrip().endswith(
            "**Full Changelog**: https://github.com/o/r/compare/v1.0.0...v1.1.0"
        )

    def test_first_release_links_to_the_commit_list_instead(self) -> None:
        # There is nothing to compare against, and a compare URL with an empty
        # base renders as a broken link on the release page.
        body = render([parse_commit("c1", "fix: x")], "v1.0.0", "", "o/r")
        assert body.rstrip().endswith(
            "**Full Changelog**: https://github.com/o/r/commits/v1.0.0"
        )
        assert "/compare/" not in body

    def test_no_repo_means_no_link_rather_than_a_broken_one(self) -> None:
        assert "Full Changelog" not in render([parse_commit("c", "fix: x")], "v1.0.0")

    def test_empty_commit_range_still_renders(self) -> None:
        # A release with no commits since the last one is degenerate but must
        # not crash the job mid-publish.
        body = render([], "v1.0.1", "v1.0.0", "o/r")
        assert "Full Changelog" in body

    def test_short_shas_are_emitted_for_github_autolinking(self) -> None:
        body = render([Commit("abc1234", "Bug Fixes", "A bug", False)], "v1.0.1")
        assert "* A bug (abc1234)" in body


class TestWorkflowBinding:
    """Bind the workflow to this module. The release job is the only caller, so
    a rename or flag change here is otherwise invisible until a release ships
    with an empty body."""

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parent.parent

    def _release_job_directives(self) -> str:
        text = (self._repo_root() / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        _, _, job = text.partition("\n  release:")
        assert job, "expected a `release:` job in ci.yml"
        return "\n".join(
            ln for ln in job.splitlines() if not ln.lstrip().startswith("#")
        )

    def test_script_exists_where_the_workflow_invokes_it(self) -> None:
        job = self._release_job_directives()
        assert "python3 release_notes.py" in job
        assert (self._repo_root() / "release_notes.py").is_file()

    def test_workflow_passes_every_argument_the_parser_requires(self) -> None:
        job = self._release_job_directives()
        for flag in ("--tag", "--previous", "--repo"):
            assert flag in job, f"release job must pass {flag}"

    def test_notes_come_from_the_file_not_generate_notes(self) -> None:
        # --generate-notes enumerates merged PRs only. This repo has never
        # opened one, so it produced a bare compare link no matter what shipped.
        # Reverting to it would silently empty every future release body.
        job = self._release_job_directives()
        assert "--notes-file" in job
        assert "--generate-notes" not in job

    def test_checkout_fetches_full_history(self) -> None:
        # read_commits runs `git log <previous>..<tag>` locally, which the
        # default depth-1 checkout cannot answer.
        assert "fetch-depth: 0" in self._release_job_directives()

    def test_plan_step_exports_the_previous_tag(self) -> None:
        job = self._release_job_directives()
        assert "previous=${previous}" in job
        assert "steps.plan.outputs.previous" in job

    def test_every_section_heading_is_unique_per_type(self) -> None:
        # SECTIONS is ordered pairs, not a dict, so a duplicated *type* would be
        # silently shadowed rather than rejected.
        types = [type_ for type_, _ in SECTIONS]
        assert len(types) == len(set(types))

    def test_repo_history_is_fully_categorised(self) -> None:
        # The whole premise is that this repo's commit subjects are already
        # well-formed enough to machine-read. If discipline slips, notes quietly
        # degrade into an "Other Changes" dump -- fail while it is still a few.
        import subprocess

        log = subprocess.run(
            ["git", "log", "-40", "--no-merges", "--pretty=format:%h\x1f%s"],
            capture_output=True,
            text=True,
            cwd=self._repo_root(),
            check=False,  # returncode is inspected below; a non-git tree is fine
        )
        if log.returncode != 0:  # not a git checkout (e.g. sdist) -- nothing to check
            return
        subjects = [ln.split("\x1f", 1) for ln in log.stdout.splitlines() if ln.strip()]
        uncategorised = [
            s for sha, s in subjects if parse_commit(sha, s).section == OTHER
        ]
        assert len(uncategorised) <= 2, (
            f"{len(uncategorised)} of the last {len(subjects)} commits are not "
            f"conventional commits and would land under {OTHER!r}: {uncategorised}"
        )
