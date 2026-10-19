# Asset provenance

## `provider_snapshot.json`

| | |
|---|---|
| Source | Six Anthropic documentation pages, read on 2026-10-17 (URLs below). Model identifiers, lifecycle states, dates, prices, context windows and cut-offs were transcribed by hand into a small JSON file. None of the pages' prose or tables is reproduced. |
| Licence | The transcription is released CC BY-NC-SA 4.0 (c) 2026 RealAI with the rest of the course. The facts themselves belong to no one; the source pages remain Anthropic's and keep their own terms. |
| Retrieved | 2026-10-17 (22:54 UTC). This is the second reading; the first was taken on 2026-09-29 and is described below. |
| Size | about 8 KB |
| Gated? | No. The pages are public; nothing to register for, and the lesson reads only the committed file. No network access is needed to run it. |

Pages:

- <https://platform.claude.com/docs/en/models/overview> (ids, context windows, cut-offs and retirement floors of the current lineup; the URL used on 2026-09-29, `/docs/en/about-claude/models/overview`, now redirects here)
- <https://platform.claude.com/docs/en/about-claude/model-deprecations> (status, deprecation and retirement dates, recommended replacements)
- <https://platform.claude.com/docs/en/about-claude/pricing> (prices for every id)
- <https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions> (what an alias is)
- <https://platform.claude.com/docs/en/models/haiku-5-5/overview> (release date, status and price tiers of Claude Haiku 5.5)
- <https://platform.claude.com/docs/en/models/haiku-4-5/overview> (Claude Haiku 4.5, which is now on a legacy page: its context window, cut-offs, status and alias)

Hand-made choices, so a reviewer can check them:

- `null` means the page does not say. The lesson treats it as unknown.
- `1M tokens` is stored as 1,000,000 and `200K tokens` as 200,000.
- `retirement_is_earliest` is true where the page says "Not sooner than".
- The pricing page names models by product name. Names were mapped to ids only where one id in the
  deprecations table carries that name: Opus 4.5 = claude-opus-4-5-20251101, Opus 4.1 =
  claude-opus-4-1-20250805, Opus 4 = claude-opus-4-20250514, Sonnet 4.5 = claude-sonnet-4-5-20250929,
  Sonnet 4 = claude-sonnet-4-20250514, Haiku 3.5 = claude-3-5-haiku-20241022, Haiku 4.5 =
  claude-haiku-4-5-20251001. Opus 5.5, 5, 4.8, 4.7 and 4.6, Sonnet 5.5, 5 and 4.6, Haiku 5.5 and the
  Fable/Mythos rows map by their dateless ids. claude-mythos-preview has no price in the pricing table,
  and claude-3-7-sonnet and claude-3-haiku are not in it.
- Claude Haiku 5.5 is priced by prompt length: $0.10 input and $0.50 output per million tokens for prompts
  up to 100,000 tokens, and $0.50 and $2.50 above that. The snapshot has one price per model, so it records
  the first tier (0.1 and 0.5) and nothing about the second. No role in `models.yaml` pins this model.
- The alias table holds the two aliases the pages document by name: claude-haiku-4-5 and
  claude-sonnet-4-5. Claude Haiku 5.5 has no separate alias: the models overview lists the same string
  for its Claude API ID and its Claude API alias.
- Where the pages give a recommended replacement, it is the id the deprecation history names for that
  model on the day of the reading.

### The first reading, 2026-09-29, and what differs

The snapshot was first taken on 2026-09-29 (22 models, from the first four pages above, the models overview
then at its old address). The provider's pages had moved by 2026-10-14, and the snapshot was re-taken from the
live pages on 2026-10-17. Apart from the rows below, every status, date, replacement and price in the first
reading was found unchanged (diffed by script), and so were the context windows and cut-offs (compared by
eye). The differences:

| Model id | First reading, 2026-09-29 | Second reading, 2026-10-17 |
|---|---|---|
| claude-sonnet-4-5-20250929 | Active; deprecated: none; retirement "not sooner than" 2026-09-29; replacement: none | Deprecated on 2026-09-30; retirement 2026-11-30 (no longer a floor); replacement claude-sonnet-5-5 |
| claude-sonnet-4-20250514 | Retired; replacement claude-sonnet-4-6 | Retired (unchanged); replacement claude-sonnet-5-5 |
| claude-3-7-sonnet-20250219 | Retired; replacement claude-sonnet-4-6 | Retired (unchanged); replacement claude-sonnet-5-5 |
| claude-haiku-5-5 | not in the table | Active; released 2026-10-07; retirement "not sooner than" 2027-10-07; 1M-token context window; reliable and training cut-offs Jun 2026; price 0.1 and 0.5 (first tier) |

The pages themselves also moved: the models overview is now at the address above, and Claude Haiku 4.5
(`claude-haiku-4-5-20251001`, unchanged facts: 200K-token context window, reliable cut-off Feb 2025,
training cut-off Jul 2025, "Not sooner than" 2026-10-15) is on its own legacy page, no longer one of the
overview's current models. The lesson's output is computed from the file, so it changed with it.

## `models.yaml`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC BY-NC-SA 4.0, (c) 2026 RealAI, part of Synapsa Commons: <https://creativecommons.org/licenses/by-nc-sa/4.0/> |
| Retrieved | 2026-09-29 (written, not fetched) |
| Gated? | No. |

A role registry that plays the part of one nobody has re-checked for a while. It is stale in several
places on purpose. Its recorded prices, limits and cut-offs are the registry's beliefs, not claims about
what a provider once charged.
