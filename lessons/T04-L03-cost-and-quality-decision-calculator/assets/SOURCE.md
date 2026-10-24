# Asset provenance

## `price_snapshot.json`

| | |
|---|---|
| Source | Public web pages, documentation and data files, read on 2026-10-02 and 2026-10-03 (URLs below). The prices, the billing structure, one licence name, the two exchange rates, two models' parameter counts and one published serving benchmark's figures were transcribed by hand into a small JSON file. None of the pages' prose is reproduced; of their tables, only the benchmark's figures, as numbers. |
| Licence | The transcription is released CC BY-NC-SA 4.0 (c) 2026 RealAI with the rest of the course. Prices, exchange rates and benchmark figures are facts; the source pages remain their owners' and keep their own terms. |
| Retrieved | 2026-10-02, re-read in full on 2026-10-24 (every price, billing rule and the exchange rates; one price changed, the hosted model's cache-hit price, 0.20 to 0.10 USD per million tokens, and the exchange rates moved), with the open teacher's price and licence, the statements on how hosts serve a fine-tuned model and Together's documented H100 figure added on 2026-10-03, and the serving benchmark (`serving_benchmark`) and the small model's size (`params`, `arch` in its Tinker row) added on 2026-10-03. Every block in the file carries `read_on`, the day its figures were read. |
| Size | about 6 KB |
| Gated? | No. Every page is public and needs no account. The lesson reads only the committed file and needs no network access to run. |

Pages:

- <https://platform.claude.com/docs/en/about-claude/pricing> (Claude Sonnet 5.5 input, output and cache-hit price, USD per million tokens; read 2026-10-02 and again 2026-10-24, when the cache-hit price had fallen from 0.20 to 0.10)
- <https://www.together.ai/pricing> and <https://www.together.ai/models/qwen3-5-397b-a17b> (the open teacher Qwen3.5-397B-A17B on Together's serverless API, input and output price per million tokens; read 2026-10-03. LoRA fine-tuning price and minimum charge for Qwen3.5 9B and the dedicated NVIDIA HGX H100 price per GPU-hour; read 2026-10-02)
- <https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/LICENSE> (the open teacher's licence, Apache-2.0; read 2026-10-03)
- <https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/> and <https://tinker-docs.thinkingmachines.ai/tinker/models.json> (Qwen3.5-9B training price and checkpoint storage price, read 2026-10-02; the beta status of Tinker's serverless inference, read 2026-10-03), and <https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/> (the sampling client that generates from trained weights, read 2026-10-03)
- <https://docs.together.ai/docs/fine-tuning/lora-vs-full> and <https://docs.together.ai/docs/dedicated-endpoints/pricing> (a LoRA fine-tune is served on a dedicated endpoint billed by the minute, not per token; the documentation's H100 figure; read 2026-10-03)
- Archived copies of Together's pricing page, <https://web.archive.org/web/20260930201344/https://www.together.ai/pricing>, and of its dedicated-inference pricing documentation, <https://web.archive.org/web/20260821030659/https://docs.together.ai/docs/dedicated-endpoints/pricing> and <https://web.archive.org/web/20260906090149/https://docs.together.ai/docs/dedicated-endpoints/pricing> (the H100 promotion that ended on 2026-09-30, and the documentation's figure before and during it; read 2026-10-03)
- <https://fireworks.ai/pricing> (embedding price for models up to 150M parameters; read 2026-10-02)
- <https://docs.fireworks.ai/faq-new/billing-pricing/are-there-extra-fees-for-serving-fine-tuned-models> and <https://docs.fireworks.ai/fine-tuning/deploying-loras> (a trained LoRA model needs a dedicated deployment; read 2026-10-03)
- <https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html> (NVIDIA NIM LLMs benchmarking guide, version 1.0.0, Performance: Llama-3.1-8B-Instruct on one H100 80GB, NIM container 1.8.0, three runs; read 2026-10-03; an archived copy of 2026-09-15 is at <https://web.archive.org/web/20260915150912/https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html>), with the same guide's <https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html> and <https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/parameters.html> (what TTFT, ITL, throughput and concurrency mean; read 2026-10-03)
- <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and <https://arxiv.org/abs/2407.21783> (the benchmark's model: 8B parameters, a dense Transformer; read 2026-10-03), and <https://huggingface.co/Qwen/Qwen3.5-9B> (the small model's layout; read 2026-10-03). The small model's 9B parameters and 'Dense' come from the Tinker row and models.json above.
- <https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml> (European Central Bank euro foreign exchange reference rates for 2026-10-23: USD and INR per euro, read 2026-10-24, a Saturday; the snapshot's first version used the rates of 2026-10-02)

Attribution, as the ECB's terms require: the exchange rates are the ECB's euro foreign exchange reference rates,
published by the European Central Bank. The ECB allows free use of the information on its site if it is reproduced
accurately and the ECB is cited as the source (<https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html>, read 2026-10-02).
The ECB publishes the rates for information only and discourages using them for transactions.

Hand-made choices, so a reviewer can check them:

- Prices are USD per million tokens unless a key says otherwise (`usd_per_gpu_hour`, `storage_usd_per_gb_month`, `minimum_charge_usd`).
- The open teacher's row on Together's pricing page also carries a cached-input price and the page has a "Batch API
  price" toggle. The snapshot holds the ordinary input and output prices with the toggle off; the cached price is not
  used because the lesson prices every teacher call without a cache.
- Tinker's table also carries prefill, sample and cached-prefill prices for the small model and, on some models, a
  limited-time discount. The small model has no discount note on the page read. Only its training price is kept: the
  lesson trains the distilled model on Tinker but serves it on a dedicated endpoint. Tinker's per-token prefill and
  sample prices are for sampling through its training service; its serverless inference, the per-token serving it
  describes, is a beta for Inkling and Inkling-Small only. Not pricing serving on Tinker is the author's assumption.
- The Together fine-tuning table has two tabs, LoRA and Full. The row used is the first table, which sits under the
  LoRA tab (checked in the page's markup).
- Together's pricing page gives 5.49 USD per GPU-hour for an on-demand H100 in its Dedicated Inference table, with
  no promotion; its dedicated-inference pricing documentation listed 3.99 in its hardware table on 2026-10-03 while using 5.49 in
  its worked example, and lists 5.49 again on 2026-10-24. 3.99 is the promotional price the pricing page showed until 2026-09-30, and the documentation's
  table changed from 5.49 to 3.99 while that promotion ran (archived copies above). The snapshot keeps 5.49 in
  `dedicated_usd_per_gpu_hour` and records the documentation's 3.99 and the promotion's end in `dedicated_docs_table`,
  which the notebook prints but does not price (see `claims.yaml`).
- The `fx` block holds units of each currency per euro, as the ECB quotes them, with `EUR` as 1.0.
- The three strings in `serving_a_fine_tune` are the author's paraphrases of the providers' statements quoted in
  `claims.yaml`; the notebook prints them as paraphrases.
- Bedrock's price page could not be read as text (its prices are filled in by script), so no Bedrock price is in the snapshot.
- `serving_benchmark` holds three of the NIM page's tables for Llama-3.1-8B-Instruct on one H100 80GB (NIM container
  1.8.0): FP8 with 200 input and 200 output tokens, FP8 with 1000 and 1000, and BF16 with 1000 and 1000 (the page's only
  16-bit run of this model on the H100 in that version), eight concurrencies each, every figure copied as printed
  (checked cell by cell against the page's HTML). The run closest to the scenario's request comes first. The page's
  throughput column is output tokens per second (its Metrics page). `model_params` is the size Meta states, 8B (the
  model page's file metadata counts 8,030,261,248); the small model's `params`, 9,000,000,000, is the `params` field of
  its entry in Tinker's models.json, which matches the 9B its card gives for its language model (the card's file
  metadata counts 9,653,104,368, a vision encoder included, which text-only requests do not use).
- The guide's latest version no longer lists the Performance page; the versioned page cited is live and archived.

## `scenario.json`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC BY-NC-SA 4.0, (c) 2026 RealAI, part of Synapsa Commons: <https://creativecommons.org/licenses/by-nc-sa/4.0/> |
| Retrieved | 2026-10-02 (written, not fetched); `teacher_model` added 2026-10-03; `serving_precision` and `peak_to_average` added, and `endpoint_capacity_requests_per_month` and the finetune, distillation and mix entries of `assumed_latency_ms` derived, 2026-10-03 |
| Gated? | No. |

An imaginary support-ticket triage task. Every value, including the choice of teacher and small model, the always-on
endpoint each small model runs on and the assumed quality and latency of each strategy, is an assumption written by
the author: none is a measurement of any model or a fact about any company. `endpoint_capacity_requests_per_month` is
the author's estimate: the notebook derives it, step by step, from the snapshot's `serving_benchmark` (a published
benchmark of a similar model) and this file's `serving_precision`, `peak_to_average`, token counts, latency budget and
endpoint hours, and the file holds the result rounded down to a multiple of 100,000. The latency of the two small
models in `assumed_latency_ms` (the same for both: one model on one GPU) is the same derivation's time for one request
on an idle endpoint, rounded up to a multiple of 10 ms, and the mix's is the average over its traffic of that and the
hosted model's assumed latency; the notebook prints both checks.
