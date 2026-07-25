# archive-data/

The actual transcripts (`*.json` + `*.metadata.json`) for every archived
video — real transcription/diarization work, not regenerable in minutes the
way the compiled search index is. Committed here as a third durable copy
(alongside the live host and `backup_aboali.sh`'s daily dated backup) after
a rebuild once wiped the live index down to 1 video with no way back except
a lucky copy found on an undocumented third host. See
[wilders-search's docs/postmortem.md](https://github.com/sayfjawad/slimme-archief-zoeker/blob/master/docs/postmortem.md)
(section 5.11) for the full story — same architecture, same lesson.

**What's deliberately *not* here:** `index.sqlite` / `embeddings.npy`. Those
are large compiled binaries (embeddings.npy alone routinely exceeds
GitHub's 100MB file limit) that `build_index.py` regenerates from exactly
these transcripts in a few minutes — see `../quickstart.sh` to do that
yourself.

Kept in sync daily by `../commit_transcripts.sh` (cron), which mirrors from
`/data/ABO_ALI` and commits only what changed.
