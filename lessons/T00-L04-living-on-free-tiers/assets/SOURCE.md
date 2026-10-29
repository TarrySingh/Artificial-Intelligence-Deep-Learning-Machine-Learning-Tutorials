# Asset provenance

## `free_tiers.json`

| | |
|---|---|
| Source | Four vendors' documentation pages, read on 2026-10-14 (URLs below; Groq contributes two pages). Limit figures were transcribed by hand into a small JSON file. None of the pages' prose or tables is reproduced. |
| Licence | The transcription is released CC BY-NC-SA 4.0 (c) 2026 RealAI with the rest of the course. The facts themselves belong to no one; the source pages remain their owners' and keep their own terms. |
| Retrieved | 2026-10-14 |
| Size | about 3 KB |
| Gated? | No. The pages are public; nothing to register for, and the lesson reads only the committed file. No network access is needed to run it. |

Pages:

- <https://console.groq.com/docs/rate-limits> (the "Free Plan Limits" table; scope; `retry-after` header; the header example, marked illustrative, whose requests-per-day reset is minutes away)
- <https://console.groq.com/docs/billing-faqs> (the Developer tier, not the Free tier, is the one that needs a payment method)
- <https://openrouter.ai/docs/api-reference/limits> (free-model request limits; its table's number cells were EMPTY when the rendered page was read in a browser, because the page's own scripts fill them, so the figures were read from its markdown source at <https://openrouter.ai/docs/api-reference/limits.md>, where the constants `FREE_MODEL_RATE_LIMIT_RPM`, `FREE_MODEL_NO_CREDITS_RPD`, `FREE_MODEL_HAS_CREDITS_RPD` and `FREE_MODEL_CREDITS_THRESHOLD` hold them; the daily figures and the threshold are corroborated by the markdown source of the FAQ, <https://openrouter.ai/docs/faq.md>, which also calls the lower daily figure a total across free models; the limits page's Tip says more accounts or API keys do not affect its rate limits and that limits differ for different models, and its key endpoint counts free-model requests per UTC day)
- <https://inference-docs.cerebras.ai/support/rate-limits> (the "Free Trial" table; token-bucket statement; the FAQ on whether a free tier renews; read in full also as <https://inference-docs.cerebras.ai/support/rate-limits.md>)
- <https://ai.google.dev/gemini-api/docs/rate-limits> (scope per project, daily reset, and the fact that the numbers are shown in AI Studio)

Hand-made choices, so a reviewer can check them:

- `null` means the page does not say. The lesson treats it as unknown, never as unlimited. The Gemini
  row has `limits: null` as a whole because the page publishes no free-tier figure at all.
- `K` in Groq's table is read as one thousand and `M` in Cerebras's table as one million. Groq's own
  header example shows `14400` for a limit its table would print as `14.4K`, which fixes the reading of K.
- One model per provider was transcribed (`openai/gpt-oss-120b` for Groq, `gpt-oss-120b` for Cerebras).
  Other rows of the same tables were not transcribed; the lesson makes no claim about them.
- Cerebras lists total and uncached tokens per minute. The row records the uncached figure (the page
  calls it the primary limit; the lesson charges every call as uncached) and says so in its `note`.
- Cerebras also lists tokens per hour. It equals the daily figure, so it can never bind first and is not
  a field of the lesson's limit type.
- `zero_dollar` is false where the source says a purchase, or a payment method, is needed to get the
  limits in the row. It is true where the source implies none is needed: Groq's billing FAQ asks for a
  payment method only to upgrade from the Free tier to the Developer tier; OpenRouter's table row for
  fewer than the threshold of credits purchased covers an account that has bought none; Google lists a
  Free usage tier qualified by an active project. It is a reading of the source page, not a prediction
  about any learner's country or account, and no page read states that any of these free allowances
  expires or when it would be withdrawn.
- OpenRouter's two rows are named for the table's threshold of 10 credits. The page grants the higher daily
  ceiling from one credit below it (9), so the lower ceiling applies below 9 credits purchased; the
  `tier` labels in the file say so. The row ids keep the table's wording.
- `scope` is the unit a page says the limit belongs to: an organisation (Groq, Cerebras), a project
  (Google) or an account (OpenRouter, whose page also says additional accounts or API keys will not
  affect its rate limits). It is never a single key on any page read.
- OpenRouter's page gives 20 requests a minute as one figure. It does not say whether that is one pool for
  every `:free` model or one per model; the lesson shares one counter across a class, the cautious reading.
  Its Tip says limits differ for different models and suggests sharing the load that way, which would help
  only if the figure is per model.
- OpenRouter's table is keyed on "credits purchased (all time)"; the lesson writes "credits" and never a
  currency, because the page does not name one.
