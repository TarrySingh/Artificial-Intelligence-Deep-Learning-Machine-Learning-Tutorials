# Asset provenance

## `provider_snapshot.json`

| | |
|---|---|
| Source | Four Anthropic documentation pages, read on 2026-09-29 (URLs below). Model identifiers, lifecycle states, dates, prices, context windows and cut-offs were transcribed by hand into a small JSON file. None of the pages' prose or tables is reproduced. |
| Licence | The transcription is released CC BY-NC-SA 4.0 (c) 2026 RealAI with the rest of the course. The facts themselves belong to no one; the source pages remain Anthropic's and keep their own terms. |
| Retrieved | 2026-09-29 |
| Size | about 8 KB |
| Gated? | No. The pages are public; nothing to register for, and the lesson reads only the committed file. No network access is needed to run it. |

Pages:

- <https://platform.claude.com/docs/en/about-claude/models/overview> (ids, aliases, context windows, cut-offs, current prices)
- <https://platform.claude.com/docs/en/about-claude/model-deprecations> (status, deprecation and retirement dates, recommended replacements)
- <https://platform.claude.com/docs/en/about-claude/pricing> (prices for the older ids)
- <https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions> (what an alias is)

Hand-made choices, so a reviewer can check them:

- `null` means the page does not say. The lesson treats it as unknown.
- `1M tokens` is stored as 1,000,000 and `200K tokens` as 200,000.
- `retirement_is_earliest` is true where the page says "Not sooner than".
- The pricing page names models by product name. Names were mapped to ids only where one id in the
  deprecations table carries that name: Opus 4.5 = claude-opus-4-5-20251101, Opus 4.1 =
  claude-opus-4-1-20250805, Opus 4 = claude-opus-4-20250514, Sonnet 4.5 = claude-sonnet-4-5-20250929,
  Sonnet 4 = claude-sonnet-4-20250514, Haiku 3.5 = claude-3-5-haiku-20241022. Opus 5, 4.8, 4.7, 4.6,
  Sonnet 5 and 4.6 and the Fable/Mythos rows map by their dateless ids. claude-mythos-preview has no
  price in the pricing table, and claude-3-7-sonnet and claude-3-haiku are not in it.
- The alias table holds the two aliases the pages document by name: claude-haiku-4-5 and
  claude-sonnet-4-5.

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
