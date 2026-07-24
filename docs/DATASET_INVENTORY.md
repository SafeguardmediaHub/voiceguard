# VoiceGuard — Dataset Inventory & Provenance

**Document ID:** VG-DOC-003
**Version:** 1.0
**Date:** 2026-07-22
**Owner:** Michael Ologungbara
**Classification:** Confidential — internal and legal counsel
**Status:** Baseline issue. **§8 contains unresolved commercial-licensing exposure that requires legal review before enterprise sale.**
**Review cycle:** On every change to a training manifest, and at minimum quarterly.
**Related:** Model Development History (VG-DOC-001) · Model Card (VG-DOC-002) · Model Inventory Register (VG-DOC-004, pending) · GRC Control & Risk Pack (VG-DOC-005, pending)

> **Scope note.** This document records what data exists, where it came from, and what is and is not known about the right to use it. §8 identifies licensing risk; it is **not legal advice**. The licence characterisations are marked as requiring verification precisely because they must be confirmed by counsel against the actual licence texts before commercial reliance.

---

## 1. Purpose

Enterprise procurement, privacy review, and ISO/IEC 42001 data-governance all begin with the same question: *what was this model trained on, and were you entitled to use it?*

This inventory answers that for every dataset that has touched a VoiceGuard model, covering:

1. **Identity and provenance** — what it is, who produced it, how it reached this project
2. **Composition** — verified file counts, languages, real/fake balance
3. **Purpose** — which model versions consumed it, and for what
4. **Personal-data status** — whether it contains identifiable human voice data
5. **Licence and consent posture** — what is known, what is assumed, and what is unverified

It supports ISO/IEC 42001 (AI system data governance, resource documentation) and ISO/IEC 27001 (information asset inventory, acceptable use).

---

## 2. Evidentiary basis

Counts marked **[verified]** were produced by enumerating the actual files on disk on 2026-07-22. Counts marked *[notebook]* come from retained training-notebook outputs with a cell citation. Counts marked *[narrative]* come from handoff prose and are the weakest class of evidence.

| Source | Type | Evidences |
|---|---|---|
| Local filesystem enumeration | **Verified** | §4 counts for all locally held datasets |
| `voiceguard-training-notebook (3).ipynb` | Retained outputs | Training corpus sizes, bucket composition |
| `generate_bias_fakes.py` | Source code | TTS engines, voice selection, manifest schema |
| `urls_*.txt` | Source lists | YouTube provenance of studio/broadcast audio |
| `build_v9_manifest.py` | Source code | Manifest construction, bucket caps, imbalance guard |
| Handoff summaries | Narrative | Historical context |

---

## 3. Dataset register — summary

| # | Dataset | Role | Scale | Personal data | Licence risk |
|---|---|---|---|---|---|
| A | ASVspoof 2019 LA | Train (real + fake) | 7,740 real *[notebook]* | Yes — human speech | 🟠 Verify |
| B | WaveFake | Train (fake) | 157,066 *[notebook]* | Derived from human speech | 🔴 **Likely non-commercial** |
| C | VCTK Corpus | Train (real) | 21,695 manifest *[notebook]* | Yes — 100+ identified speakers | 🟠 Verify |
| D | Common Voice — Kaggle repackage | Train (real) | *[narrative]* | Yes — crowd-sourced | 🟢 Likely CC0 |
| E | Common Voice v26 — official | Bias audit (real) | **27,929** **[verified]** | Yes — crowd-sourced | 🟢 Likely CC0 |
| F | "In-the-Wild" | Train (real + fake) | 500 + 500 *[notebook]* | Yes — public figures | 🟠 Verify |
| G | Bark / XTTS / ElevenLabs outputs | Train (fake) | 1,177 *[notebook]* | Synthetic | 🔴 **ToS + XTTS non-commercial** |
| H | noiz.ai clones | Train + held-out (fake) | 116 *[narrative]* | Synthetic clone of a voice | 🟠 Verify |
| I | Studio/broadcast clips | Train (real) + FP study | **494** **[verified]** | Yes — identifiable speakers | 🔴 **YouTube — no licence** |
| J | Bias-audit TTS fakes | Bias audit | **599** **[verified]** | Synthetic | 🟠 TTS provider ToS |
| K | edge-tts / gTTS generated | Train (fake) | 873 dir total **[verified]** | Synthetic | 🟠 ToS |
| L | Golden regression clips | Test fixture | 4 **[verified]** | Yes | Inherits source |

**Legend:** 🟢 low concern · 🟠 requires verification · 🔴 **material exposure for a commercial product**

---

## 4. Dataset detail sheets

### A. ASVspoof 2019 — Logical Access

| Field | Value |
|---|---|
| **Type** | Academic anti-spoofing benchmark |
| **Role** | Foundational training corpus, V1 onward; also the `fake_asvspoof_2019` bucket (500) |
| **Scale** | 7,740 real utterances *[notebook 100]* |
| **Provenance** | Public research release, obtained via Kaggle |
| **Personal data** | Yes — recorded human speech |
| **Licence** | ⚠️ **Requires verification.** ASVspoof releases are distributed for research; commercial-use terms must be confirmed. |

**Note of significance:** ASVspoof 2019 has been a *training* corpus since V1, while ASVspoof 2021 is the intended *evaluation* benchmark (VG-DOC-001 §16.4, never run). Using one release for training and a sibling release for headline evaluation is defensible only if the split is stated plainly. It must not be presented as an independent benchmark without that disclosure.

**Spectral characteristic that shaped the project:** ASVspoof reals measured spectral flatness 0.0105 against VCTK's 0.0164 *[notebook 96–99]* — unusually "clean". This narrow real-audio distribution is the origin of the "clean = fake" proxy that produced the 65.5% studio false-positive rate (VG-DOC-001 §7).

---

### B. WaveFake

| Field | Value |
|---|---|
| **Type** | Academic vocoder-artifact dataset (6 neural vocoders) |
| **Role** | Primary fake corpus from V1; `fake_wavefake` bucket (500) |
| **Scale** | **157,066 files** *[notebook 100]* |
| **Personal data** | Derived — vocoded from LJSpeech/JSUT source speech |
| **Licence** | 🔴 **Requires verification — likely non-commercial.** |

**The class-imbalance origin.** WaveFake (157,066 fake) against ASVspoof (7,740 real) is roughly **20:1**. That imbalance is the root of the failure that collapsed AASIST V9 four versions later (VG-DOC-001 §12.2, defect 1). It is a property of the *data*, not merely of one training script, and any future retrain must handle it explicitly.

---

### C. VCTK Corpus

| Field | Value |
|---|---|
| **Type** | Multi-speaker English corpus (University of Edinburgh) |
| **Role** | Added during V1–V3 to broaden the real distribution |
| **Scale** | 21,695-file manifest — mic1 only, capped 200/speaker *[notebook 99]* |
| **Personal data** | Yes — 100+ consented, individually identified speakers |
| **Licence** | 🟠 Commonly released under an open licence (CC BY 4.0 for v0.92) — **verify the version and text held.** |

**Why it was added — and why it matters.** Spectral census showed VCTK had higher flatness than ASVspoof, and the notebook records the reasoning verbatim: *"VCTK has higher flatness than ASVspoof — good. Training on VCTK will teach the model to accept [clean real speech]"* *[notebook 99]*.

This is the **first** attempt to fix "clean = fake". It worked partially and the failure mode returned at scale in V8. Documented here because it shows the problem was understood early — and that a data-composition fix alone did not durably solve it.

---

### D & E. Common Voice (Mozilla) — two distinct holdings

**D. Kaggle repackage (training).** Used for the `real_common_voice` bucket. The Phase 6 handoff records this as *"a flattened English-only Kaggle repackage (vedant2022), not the official Mozilla multilingual structure. No demographic metadata TSV available."*

**The absence of demographic metadata is a governance limitation, not a footnote:** without speaker age/gender/accent fields, demographic fairness analysis on the *training* distribution is impossible. The Phase 6 bias audit measured fairness across *languages*, not across demographic groups within a language — a narrower claim than "bias audited" implies.

**E. Official Common Voice v26 (bias audit) — [verified]:**

| Language | Clips |
|---|---|
| Hausa (`ha`) | 10,450 |
| Igbo (`ig`) | 12,553 |
| Yoruba (`yo`) | 4,926 |
| **Total** | **27,929** |

**This supersedes a documented blocker.** The Phase 6 handoff recorded multilingual sourcing as blocked. The official corpus with all three major Nigerian languages is now held locally at `cv-corpus-26.0-2026-06-12/`. **It is largely unexploited** — the bias audit used only 50 reals per language. With ~28,000 Nigerian-language clips available, both the untested Yoruba/Igbo catch rate (VG-DOC-001 §17 item 5) and the thin real-audio coverage could be materially improved.

**Licence:** Common Voice is released CC0 — the lowest-risk holding in this inventory. **Verify against the v26 release text.**

---

### F. "In-the-Wild"

| Field | Value |
|---|---|
| **Role** | `real_itw` (500) and `fake_itw` (500) buckets |
| **Personal data** | Yes — reportedly public figures / politicians |
| **Licence** | 🟠 **Requires verification.** |

**Operationally the hardest real bucket.** `real_in_the_wild` carries the highest false-positive rate of any real source — **30.3–31.7%** *[notebook 419–420]*, driven by the un-retrained RawNet3 (VG-DOC-001 §17 item 3). It is also the reason the bias-audit set is "easier" than the validation set and why the 0.00% cascade EER must never be quoted (VG-DOC-001 §11.3).

**Elevated privacy consideration:** if the speakers are public figures, the recordings are identifiable voice data of named individuals used to train a commercial biometric-adjacent system. That warrants specific counsel attention regardless of the dataset's stated licence.

---

### G. Commercial and open TTS generator outputs

Verified V7/V8-era bucket composition *[notebook 253]*:

| Bucket | Count | Generator |
|---|---|---|
| `fake_bark` | 282 | Bark (Suno) |
| `fake_bark_opus` | 282 | Bark → Opus transcode |
| `fake_xtts` | 250 | XTTS (Coqui) |
| `fake_xtts_opus` | 250 | XTTS → Opus transcode |
| `fake_elevenlabs_mp3` | 56 | ElevenLabs |
| `fake_elevenlabs_opus` | 57 | ElevenLabs → Opus |

**The `_opus` variants are deliberate codec simulation** — Opus is what WhatsApp and most VoIP use. Paired with the `real_whatsapp_opus` bucket (571), this is a considered attempt to cover messaging-app audio, the dominant real-world channel for voice fraud.

**Licence exposure — 🔴 material:**

- **XTTS (Coqui)** is distributed under the **Coqui Public Model Licence**, which is **non-commercial**. Training a commercial product on its outputs requires explicit verification.
- **ElevenLabs** terms restrict use of generated audio; using outputs to train a competing or derivative model is commonly prohibited.
- **Bark (Suno)** is more permissive but the output terms still require confirmation.

**This is not a theoretical concern.** These outputs are baked into the weights of a model being sold commercially.

---

### H. noiz.ai voice clones

| Field | Value |
|---|---|
| **Role** | 70 training + 30 held-out + 16 drift-baseline fakes |
| **Provenance** | Outputs of a commercial voice-cloning service |
| **Personal data** | Synthetic — but cloned *from* a real voice |
| **Licence** | 🟠 Governed by the provider's ToS — **unverified** |

**Two governance questions:** (1) do noiz.ai's terms permit using generated audio to train a detector; and (2) **whose voice was cloned?** If a real person's voice was cloned to produce training data, their consent posture is a live question under any biometric-data regime.

This bucket carries disproportionate weight in the project's reporting — the noiz.ai catch rate (60% → 83.3%) is a headline V9 success metric, measured on **30 held-out clips**. A single-provider, 30-sample basis is thin for a headline claim and should be stated as such.

---

### I. Studio / broadcast clips — **highest licensing exposure**

**Verified composition (494 files):**

| Category | Files | Purpose |
|---|---|---|
| `podcast` | 125 | FP study + training |
| `audiobook` | 100 | FP study + training |
| `broadcast_news` | 100 | FP study + training |
| `arabic` | 50 | Bias-audit reals |
| `french` | 50 | Bias-audit reals |
| `pidgin` | 50 | Bias-audit reals |

The 325 clips in the first three categories are the set that produced the **65.5% false-positive discovery** (VG-DOC-001 §7) and, as `real_studio` (226 capped), the fix.

**Provenance: downloaded from YouTube via `yt-dlp`.** The source URLs are committed in the repository — `urls_audiobook.txt`, `urls_news.txt`, `urls_podcast.txt`, `urls_arabic.txt`, `urls_french.txt`, `urls_pidgin.txt`.

**🔴 This is the most serious licensing exposure in the inventory, on three independent grounds:**

1. **Platform terms.** YouTube's Terms of Service prohibit downloading content absent an explicit platform-provided mechanism. `yt-dlp` is not such a mechanism.
2. **Copyright.** Podcasts, audiobooks, and broadcast news are third-party copyrighted works. No licence was obtained. Whether model training constitutes fair use/fair dealing is unsettled and jurisdiction-dependent — and Nigeria, the EU, and the US differ.
3. **Personal data.** The recordings are identifiable human voices. The speakers did not consent to their voices being processed to build a commercial detection system.

**Aggravating factor:** these clips are not incidental. They are `real_studio` — the bucket introduced specifically to fix the project's most serious defect. **Removing them would regress studio false-positive performance from 12.0% back toward 80%.** The dependency is structural, not cosmetic.

**Mitigating factor:** the *raw audio* is not distributed — it is not in git (`.gitignore` excludes media) and not in the Docker image (`.dockerignore` excludes `*.mp3`/`*.wav`). Exposure is confined to the training act and to whatever the weights are held to embody. **The URL lists, however, are committed** and constitute a written record of the sourcing method.

**Required action (§9 item 1).** Counsel must assess. If the position is untenable, the realistic remediation is to re-source equivalent clean real audio under a permissive licence — Common Voice (E, CC0), LibriVox public-domain audiobooks, or licensed broadcast material — and retrain `real_studio` from it. Common Voice v26 is already held and would cover part of this.

---

### J. Bias-audit TTS fakes — **[verified], 599 files**

Generated locally by `generate_bias_fakes.py`.

| Language | Real | Fake | Total |
|---|---|---|---|
| Arabic | 50 | 50 | 100 |
| English | 0 | 50 | 50 |
| French | 50 | 50 | 100 |
| Hausa | 50 | **100** | 150 |
| Igbo | 50 | **0** | 50 |
| Pidgin | 50 | 50 | 100 |
| Yoruba | 49 | **0** | 49 |
| **Total** | **299** | **300** | **599** |

The 599 total reconciles exactly with the reported bias-audit set size.

**Generators:** **edge-tts** (Microsoft Azure voices) with **gTTS** (Google) as fallback and for Yoruba. The manifest records `path/label/language/tts_engine/voice/gender` per entry — good provenance hygiene, and it enables per-engine and per-gender analysis that has not yet been performed.

**Three structural limitations this table makes plain:**

1. **Igbo and Yoruba have zero fakes** — no engine produced them. Catch rate for the two largest Nigerian languages by speaker population is **untested**. The "catch-rate parity PASS" therefore rests on 5 of 7 languages.
2. **Pidgin fakes are not really Pidgin.** `generate_bias_fakes.py` notes: *"No dedicated Pidgin voices — use Nigerian English"*. Pidgin's 0% FPR and its parity result are measured against **Nigerian-English TTS**, not Nigerian Pidgin synthesis.
3. **Hausa's 100 fakes are split 50 test / 50 train** — the first 50 alphabetically are the test set, the last 50 were added to training as the bias mitigation. The split is deterministic and documented, but it means Hausa's post-mitigation 100% catch is measured on a set drawn from the *same generation run* as its training fakes. Same engine, same voices, same session. **That is a generous evaluation and should be labelled as such.**

**Known data-quality defect:** 25 of the 50 Hausa test fakes fail to decode (corrupt MP3), consistently across sessions. Hausa catch-rate results rest on 25 samples.

**Licence:** edge-tts and gTTS access Microsoft and Google services through unofficial interfaces. Both providers' terms restrict automated access and downstream use of outputs. 🟠 **Requires verification.**

---

### K. Locally generated training audio — 873 files **[verified]**

`data/` — `edge_tts_output/` (en, fr, hi), `held_out/` (fake+real), `new_samples/` (fake+real), `noiz_phone/`. Same ToS considerations as J. The `noiz_phone` set underlies the phone-effect vulnerability finding (40% catch, n=5) — **a 5-sample basis for a disclosed product limitation.**

---

### L. Golden regression fixtures — 4 files **[verified]**

`tests/golden_clips/`: `real_studio_037.mp3`, `real_studio_055.mp3`, `fake_noizai_a4cd.mp3`, `fake_concert_hall.mp3`. Committed to git and shipped in the Docker image (an explicit `.dockerignore` exception) because the startup smoke check depends on one of them.

**Note:** two are `real_studio` — i.e. YouTube-derived (I) — and **are distributed**, in the repository and in every built image. This is the one place the §I exposure leaves the training environment. Four clips is a small surface, but it is not zero and should be included in the counsel assessment.

---

## 5. Derived manifests

| Manifest | Entries | Composition |
|---|---|---|
| `train_v8_fresh.json` | 4,400–4,438 | 7 source buckets |
| **`train_v9.json`** | **4,746** | V8 4,400 + `real_studio` 226 + `fake_noizai` 70 + `fake_hausa_tts` 50 |
| `val_v8_fresh.json` | 1,200 | Validation through V9 |
| `eval_v9_heldout.json` | 105 | 75 studio reals + 30 noiz.ai fakes, never trained on |
| `teacher_scores_v9_train.json` | 4,720 | V9 ensemble soft labels (26 skipped) |
| `teacher_scores_v9_val.json` | 1,199 | V9 ensemble soft labels (1 skipped) |

V7/V8 bucket composition *[notebook 253]* — real 1,873 / fake 2,677:

| Real | Count | Fake | Count |
|---|---|---|---|
| `real_clean_speech` | 802 | `fake_itw` | 500 |
| `real_whatsapp_opus` | 571 | `fake_asvspoof_2019` | 500 |
| `real_itw` | 500 | `fake_wavefake` | 500 |
| | | `fake_bark` / `_opus` | 282 / 282 |
| | | `fake_xtts` / `_opus` | 250 / 250 |
| | | `fake_elevenlabs_mp3` / `_opus` | 56 / 57 |

**A control worth crediting:** `build_v9_manifest.py` enforces per-bucket caps and emits *"⚠ WARNING: Class imbalance > 2:1"*. The manifest layer learned the lesson the raw corpora taught (§B) — even though the AASIST V9 *training script* then ignored the per-sample `weight` field the manifest carried (VG-DOC-001 §12.2, defect 3). **The data-governance control existed; the training code did not honour it.**

---

## 6. Personal-data assessment

**Voice recordings are personal data.** Under NDPR/Nigeria DPA 2023, GDPR, and most modern regimes, a voice recording identifies a natural person. Where processed *for the purpose of uniquely identifying* someone, it is biometric data and attracts a higher bar.

**VoiceGuard's position:** it determines whether audio is *synthetic*, not *who is speaking*. `MODEL_CARD.md` §2 explicitly places speaker identification and voice-biometric matching out of scope. That is a genuine and defensible distinction — the system does not perform biometric identification.

**But the training data is another matter.** Datasets A, C, D, E, F, I and L contain identifiable human speech. The relevant questions for the policy author:

| Question | Status |
|---|---|
| Lawful basis for processing the training data? | **Not documented** |
| Was any data subject informed? | Datasets C, D, E — yes, by their programmes. **I — no.** |
| Can a data subject request erasure from the training set? | **No mechanism exists** |
| Can a data subject be identified from the weights? | Not assessed; no membership-inference testing performed |
| Is production audio retained? | **No** — see §7 |

---

## 7. Production data handling — the strong part of the story

In contrast to the training-data position, the **runtime** posture is clean and defensible:

| Property | Implementation |
|---|---|
| **Uploads deleted after processing** | `worker.py` unlinks the input file in a `finally` block — deletion occurs even if detection raises |
| **No audio retained** | Only the verdict and metadata persist in `jobs.db` |
| **No training on customer data** | No pipeline ingests production audio into any manifest |
| **Transient storage only** | Uploads live in the `/data` volume between submission and processing |
| **Encrypted backups** | `jobs.db` + `auth_keys.json` only, Fernet-encrypted — **no audio** |
| **Size cap** | 25 MiB, enforced at proxy and application |
| **Access control** | SHA-256-hashed bearer keys; job results scoped to the issuing key |

**This is a strong data-minimisation story and should be stated prominently in customer-facing material.** A customer's audio is processed and deleted; it never becomes training data.

**Gaps:** no formally documented retention *policy* (behaviour is correct but unwritten — Phase 7 has this as an open task); `jobs.db` retains job metadata indefinitely with no rotation; and the audit log is not yet wired into the live `/detect` path.

---

## 8. Licensing risk register — **requires legal review**

Ordered by exposure. This is the section the GRC policy author most needs.

| # | Risk | Datasets | Severity | Why |
|---|---|---|---|---|
| 1 | **Unlicensed copyrighted material in training** | I (494 files) | 🔴 **High** | YouTube ToS + third-party copyright + no speaker consent. Structural: removing it regresses the flagship studio-FP fix. |
| 2 | **Non-commercial licence used commercially** | B, G (XTTS) | 🔴 **High** | Research/NC datasets are commonly restricted to non-commercial use. Product is sold commercially. |
| 3 | **TTS provider ToS on generated outputs** | G, J, K | 🟠 Medium | ElevenLabs/Microsoft/Google terms may prohibit training derivative models on outputs. |
| 4 | **Research-benchmark commercial terms unverified** | A, C, F | 🟠 Medium | Standard for academic corpora; needs confirmation per licence text. |
| 5 | **Cloned-voice consent unknown** | H | 🟠 Medium | Whose voice was cloned, and under what consent? |
| 6 | **No training-data erasure mechanism** | All human-speech sets | 🟠 Medium | A valid erasure request could not currently be honoured. |
| 7 | **Training-derived audio distributed** | L (2 of 4 clips) | 🟡 Low | Ships in repo and image; small surface but non-zero. |

**Honest framing for the policy author.** Items 1 and 2 are the kind of finding that stops an enterprise procurement or a due-diligence round. They are **remediable** — the remediation is re-sourcing and retraining, not redesign — but remediation costs a retrain cycle and a re-measurement of every metric in VG-DOC-001. **The cost rises with every month the current weights stay in production**, because more customers rely on the measured behaviour.

The team's own instinct here has been sound: the studio clips were sourced to fix a real and serious defect, and the fix worked. The gap is that provenance and licensing were never assessed alongside the technical decision. That is an ordinary failure mode for a research-led project entering commercial deployment — and it is exactly what this inventory exists to surface.

---

## 9. Required actions

| # | Action | Owner | Priority |
|---|---|---|---|
| 1 | **Legal review of §8 items 1 and 2** before any enterprise sale | Counsel | **Critical** |
| 2 | Obtain and file the actual licence text for A, B, C, D, E, F | Project owner | High |
| 3 | Review ToS for ElevenLabs, XTTS, edge-tts/Azure, gTTS, noiz.ai | Counsel | High |
| 4 | Plan a re-sourced `real_studio` from permissively licensed audio (Common Voice v26 is held) | ML | High |
| 5 | Write the formal data-retention policy documenting §7 behaviour | GRC | High |
| 6 | Exploit the 27,929 held Common Voice clips: Yoruba/Igbo coverage, thicker real sets | ML | Medium |
| 7 | Source or generate Yoruba and Igbo fakes to close the catch-rate gap | ML | Medium |
| 8 | Regenerate the 25 corrupt Hausa test fakes | ML | Medium |
| 9 | Re-run the bias audit with train/test fakes from **separate generation runs** (§J item 3) | ML | Medium |
| 10 | Record demographic metadata where available; assess within-language fairness | ML | Medium |
| 11 | Assess membership-inference risk on the weights | ML | Low |
| 12 | Define a data-subject erasure procedure for training data | GRC | Low |

---

## 10. Document change log

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-07-22 | Michael Ologungbara | Baseline. All locally held datasets enumerated and verified on disk. Licensing risk register established; §8 items 1–2 raised for legal review. Documented: official Common Voice v26 (27,929 Nigerian-language clips) held but largely unused; Pidgin fakes generated with Nigerian-English voices; Hausa train/test fakes from a single generation run. |
