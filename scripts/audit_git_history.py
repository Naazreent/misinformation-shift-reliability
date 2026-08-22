#!/usr/bin/env python3
"""Audit every reachable commit/blob before public release.

The report never stores matched secret or personal-data values. Findings use
categories, object identifiers, and repository paths only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree


SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "password_assignment": re.compile(
        r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"\n]{6,}"
    ),
}
PERSONAL_PATTERNS = {
    "student_identifier": re.compile(r"(?i)\bW\d{8}\b"),
    "dissertation_filename": re.compile(r"(?i)SS_W\d+\.docx"),
    "phone_number": re.compile(r"(?<!\d)(?:\+?44\s?\d{10}|0\d{10})(?!\d)"),
}
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ALLOWED_EMAIL_SUFFIX = "@users.noreply.github.com"
FORBIDDEN_HISTORY_NAMES = {"ifnd.csv", ".env"}
FORBIDDEN_HISTORY_NAME_PATTERNS = [re.compile(r"(?i)^ss_w\d+\.docx$")]


def git(repository: Path, *arguments: str, text: bool = True):
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=text, stderr=subprocess.DEVNULL
    )


def document_words(path: Path) -> tuple[str, list[str]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    words: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            root = ElementTree.fromstring(archive.read(name))
            for element in root.iter():
                if element.tag.endswith("}t") and element.text:
                    words.extend(re.findall(r"\b\w+\b", element.text.casefold()))
    return digest, words


def shingle_set(words: list[str], width: int) -> set[tuple[str, ...]]:
    if len(words) < width:
        return set()
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--reference-docx")
    parser.add_argument("--output", default="reports/git_history_audit.json")
    parser.add_argument("--max-text-bytes", type=int, default=2_000_000)
    parser.add_argument("--shingle-words", type=int, default=20)
    args = parser.parse_args()

    repository = Path(args.repository).resolve()
    commits = git(repository, "rev-list", "--reverse", "--all").splitlines()
    commit_rows = []
    findings: list[dict[str, str]] = []
    for commit in commits:
        metadata = git(
            repository,
            "show",
            "-s",
            "--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%s",
            commit,
        ).rstrip("\n").split("\x00")
        sha, author, author_email, committer, committer_email, subject = metadata
        commit_rows.append(
            {
                "sha": sha,
                "author": author,
                "committer": committer,
                "subject": subject,
                "email_policy": "github_noreply"
                if author_email.endswith(ALLOWED_EMAIL_SUFFIX)
                and committer_email.endswith(ALLOWED_EMAIL_SUFFIX)
                else "review_required",
            }
        )
        for role, email in (("author", author_email), ("committer", committer_email)):
            if not email.endswith(ALLOWED_EMAIL_SUFFIX):
                findings.append(
                    {
                        "category": "non_noreply_commit_email",
                        "object": sha,
                        "path": role,
                    }
                )

    reference_digest = None
    reference_shingles: set[tuple[str, ...]] = set()
    if args.reference_docx:
        reference_digest, words = document_words(Path(args.reference_docx))
        reference_shingles = shingle_set(words, args.shingle_words)

    objects = git(repository, "rev-list", "--objects", "--all").splitlines()
    blob_rows = []
    scanned_text = 0
    binary_blobs = 0
    oversized_blobs = 0
    for row in objects:
        sha, _, path = row.partition(" ")
        if git(repository, "cat-file", "-t", sha).strip() != "blob":
            continue
        size = int(git(repository, "cat-file", "-s", sha).strip())
        blob_rows.append((sha, path, size))
        basename = Path(path).name
        if basename.casefold() in FORBIDDEN_HISTORY_NAMES or any(
            pattern.match(basename) for pattern in FORBIDDEN_HISTORY_NAME_PATTERNS
        ):
            findings.append(
                {"category": "forbidden_history_filename", "object": sha, "path": path}
            )
        if size > args.max_text_bytes:
            oversized_blobs += 1
            continue
        payload = git(repository, "cat-file", "blob", sha, text=False)
        if b"\x00" in payload[:8192]:
            binary_blobs += 1
            continue
        text = payload.decode("utf-8", errors="replace")
        scanned_text += 1
        for category, pattern in {**SECRET_PATTERNS, **PERSONAL_PATTERNS}.items():
            if pattern.search(text):
                findings.append({"category": category, "object": sha, "path": path})
        emails = EMAIL_PATTERN.findall(text)
        if any(not email.casefold().endswith(ALLOWED_EMAIL_SUFFIX) for email in emails):
            findings.append(
                {"category": "non_noreply_email_in_blob", "object": sha, "path": path}
            )
        if reference_shingles:
            words = re.findall(r"\b\w+\b", text.casefold())
            if any(
                tuple(words[index : index + args.shingle_words]) in reference_shingles
                for index in range(max(0, len(words) - args.shingle_words + 1))
            ):
                findings.append(
                    {
                        "category": f"reference_overlap_{args.shingle_words}_words",
                        "object": sha,
                        "path": path,
                    }
                )

    unique_findings = sorted(
        {tuple(sorted(finding.items())) for finding in findings},
        key=lambda item: dict(item).get("path", ""),
    )
    report = {
        "status": "passed" if not unique_findings else "failed",
        "scope": "all reachable commits and blobs",
        "commit_count": len(commits),
        "blob_count": len(blob_rows),
        "text_blobs_scanned": scanned_text,
        "binary_blobs_skipped_from_text_scan": binary_blobs,
        "oversized_blobs_skipped_from_text_scan": oversized_blobs,
        "commit_metadata": commit_rows,
        "checks": {
            "secret_patterns": sorted(SECRET_PATTERNS),
            "personal_patterns": sorted(PERSONAL_PATTERNS),
            "commit_email_policy": "GitHub noreply addresses only",
            "forbidden_history_names": sorted(FORBIDDEN_HISTORY_NAMES),
            "reference_overlap_words": args.shingle_words if reference_shingles else None,
            "reference_docx_sha256": reference_digest,
        },
        "findings": [dict(item) for item in unique_findings],
    }
    output = repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    if unique_findings:
        raise SystemExit(f"Git history audit failed with {len(unique_findings)} finding(s)")


if __name__ == "__main__":
    main()
