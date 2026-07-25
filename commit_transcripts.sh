#!/usr/bin/env bash
# Mirrors abo-ali-search's source transcripts (*.json/*.metadata.json in
# /data/ABO_ALI) into archive-data/ inside this repo and commits+pushes any
# change -- a third durable copy of the transcripts, alongside the live
# host and backup_aboali.sh's dated backups, after a rebuild once wiped the
# live index with nothing to restore from except a lucky copy found on
# another host (see the wilders-search sibling repo's docs/postmortem.md).
# Deliberately excludes index.sqlite/embeddings.npy: large binary blobs
# that regenerate from these transcripts in minutes via build_index.py, and
# embeddings.npy alone exceeds GitHub's 100MB single-file limit.
set -u
cd "$(dirname "$0")"
LOG=/data/ABO_ALI/commit_transcripts.log
exec >> "$LOG" 2>&1
echo "=== commit_transcripts $(date '+%F %T')"

mkdir -p archive-data
rsync -a --delete --include='*.json' --exclude='*' /data/ABO_ALI/ archive-data/

git add archive-data/
if git diff --cached --quiet; then
  echo "  geen wijzigingen"
else
  N=$(git diff --cached --stat | tail -1)
  if ! git commit -q -m "Daily transcript sync: $(date '+%F')

$N

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"; then
    echo "  commit FAALDE (zie output hierboven, bv. ontbrekende git user.email/name)"
    exit 1
  fi
  git push && echo "  gecommit en gepusht" || echo "  push FAALDE"
fi
