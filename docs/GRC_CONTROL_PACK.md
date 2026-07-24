# VoiceGuard — GRC Control & Risk Pack

**Document ID:** VG-DOC-005
**Version:** 1.0
**Date:** 2026-07-22
**Owner:** Michael Ologungbara
**Classification:** Confidential — internal, GRC, legal counsel
**Status:** Baseline issue. **Two risks (R1, R2) should be treated as blocking for enterprise sale pending legal review.**
**Review cycle:** Quarterly, and on any material change to the system, its data, or its deployment.
**Related:** Model Development History (VG-DOC-001) · Model Card (VG-DOC-002) · Dataset Inventory (VG-DOC-003) · Model Inventory Register (VG-DOC-004)

---

## 0. How to use this document

**Audience:** the person writing VoiceGuard's GRC policy. This pack is the input to that work, not the policy itself.

It gives you five things:

| § | What it gives you | Use it to |
|---|---|---|
| §1–3 | System description, data flows, responsibility boundary | Write the scope statement |
| §4 | Consolidated risk register, scored | Write the risk-treatment plan |
| §5 | Controls that **already exist**, with evidence pointers | Claim credit — and prove it in an audit |
| §6 | Controls that **do not exist** | Write the remediation plan |
| §7 | The specific policies you need to author | Work through as a checklist |

**Two framing points before you start.**

First: **this system's engineering controls are stronger than its governance controls.** There is a hash-chained model registry, tamper-evident promotion logging, deterministic inference, fail-closed startup, and integrity verification — capabilities many larger vendors lack. What is missing is the documentation, ownership, and legal groundwork that turns those into an auditable management system. That is a good position to be in: the hard engineering is done.

Second: **the honest findings in §4 are an asset, not a liability.** They were surfaced by internal review before a customer or regulator found them. An organisation that can produce a register like this is demonstrating exactly the competence ISO/IEC 42001 asks for. Do not soften them — the value is in their being written down and owned.

---

## 1. System description for GRC purposes

**What it is.** VoiceGuard is an audio deepfake / synthetic-speech detector, delivered as an authenticated HTTP API. It accepts an audio file and returns a calibrated probability of synthesis, a plain-language verdict, and a forensic explainability payload.

**What it does *not* do.** It does not identify speakers, perform voice-biometric matching, or determine who is speaking or their intent. This distinction is load-bearing for privacy classification (§3.3) and is stated in the model card.

**How it decides.** A two-stage cascade per 4-second chunk:

1. **Stage 1 — LCNN screener** (CPU, ~11 ms). Confident chunks resolve here — roughly **86% of traffic**.
2. **Stage 2 — three-model ensemble** (AASIST + Wav2Vec2 + RawNet3 → XGBoost fusion → Platt calibration) for ambiguous chunks.

Output bands: `auto_fake ≥ 0.85` · `likely_fake ≥ 0.55` · `to_review ≥ 0.30` · else `auto_real`.

**Deployment.** Single DigitalOcean droplet, Docker Compose, three containers (Caddy → API + worker). VPC-private; no public exposure. Model baked into a SHA-tagged image. See the deployment design and runbook.

### 1.1 Data flow

```
Customer backend ──HTTPS+Bearer──▶ Caddy ──▶ API ──▶ /data volume ──▶ Worker
                                                          │              │
                                                    SQLite queue    detector.detect()
                                                          │              │
                                                          ▼              ▼
                                                    verdict stored   INPUT FILE DELETED
                                                          │
                                          Customer polls GET /jobs/{id}
```

**Critical property:** the uploaded audio is deleted by the worker in a `finally` block — deletion occurs even if detection raises. **No customer audio is retained, and none is ever used for training.** This is the strongest privacy fact about the system and should be stated prominently in every customer-facing document.

### 1.2 Data at rest

| Store | Contents | Retention | Encryption |
|---|---|---|---|
| `/data` uploads | Transient audio | Deleted after processing | Volume-level |
| `jobs.db` | Job metadata + verdicts. **No audio.** | Indefinite — **no rotation policy (G12)** | Volume-level |
| `auth_keys.json` | SHA-256 key hashes. No plaintext keys. | Until revoked | Volume-level |
| Spaces backups | `jobs.db` + `auth_keys.json`. **No audio.** | Nightly | Fernet (opt-in — **must be enabled**) |
| Model bundle | Weights | Immutable per image | SHA-256 integrity-verified |

---

## 2. Governance scope

**In scope:** the VoiceGuard detection service, its models, its training data and pipeline, its deployment infrastructure, and the vendor-side processes that maintain them.

**Out of scope:** the customer's application, their decisioning logic, their lawful basis for processing their own audio, their human-review process, and their end-user relationships. §3 draws that line precisely.

---

## 3. Responsibility boundary — vendor vs. customer

VoiceGuard is sold **as a platform**: the API is general-purpose and customers apply it to their own use cases (fraud/KYC, media verification, evidence triage). The boundary must be explicit in contract and documentation, because the regulatory obligations differ sharply on each side.

### 3.1 Vendor responsibilities

| Area | Commitment |
|---|---|
| Detection quality | Maintain and measure model performance; disclose limitations honestly |
| Transparency | Model card, measured metrics, known failure modes, bias-audit results |
| Data minimisation | Delete customer audio after processing; never train on it |
| Security | Authenticated access, encrypted transport, integrity-verified models |
| Change control | Versioned, hash-chained model registry with rollback |
| Notification | Inform customers of material model changes and newly discovered limitations |

### 3.2 Customer responsibilities

| Area | Obligation |
|---|---|
| Lawful basis | Establishing the legal ground for processing the audio they submit |
| Consent / notice | Informing their data subjects |
| **Threshold selection** | Choosing operating thresholds for their risk appetite (see the asymmetric-thresholds document) |
| **Human oversight** | Ensuring a human decides — the verdict is **advisory, never determinative** |
| Appeal / redress | Providing a route to contest an adverse outcome |
| Use-case fit | Confirming the system suits their audio conditions (see §3.4) |

### 3.3 Privacy classification — the argument to make

Voice recordings are personal data under NDPR / Nigeria DPA 2023, GDPR and comparable regimes. Where processed **for the purpose of uniquely identifying a natural person**, they are *biometric data* and attract a higher bar.

**VoiceGuard's defensible position:** it classifies audio as synthetic or genuine. It does not identify, match, or verify speakers. It therefore processes personal data but does **not** perform biometric identification.

**Caveat the policy must carry:** the *customer's* use may be biometric in effect. A KYC-liveness deployment sits inside an identity-verification flow, and that flow is biometric even though VoiceGuard's component is not. **The customer's obligations may exceed the vendor's.** Say so explicitly.

### 3.4 Conditions the customer must be told about

These are measured limitations, not disclaimers. They belong in the contract and in onboarding.

| Condition | Measured behaviour | Implication |
|---|---|---|
| Deliberate adversarial manipulation | Ensemble EER degrades to ~0.48 under PGD | Not suitable as a sole control against a sophisticated attacker |
| Phone / call-quality audio | 40% catch rate (n=5) | Telephony use requires customer-side validation |
| Arabic broadcast audio | 16% false-positive rate | Threshold tuning needed; costs ~10 pp of catch rate |
| Yoruba / Igbo synthetic speech | **Untested** — no test fakes existed | No catch-rate claim can be made |
| Nigerian Pidgin | Tested against **Nigerian-English** TTS | Parity result does not cover true Pidgin synthesis |

---

## 4. Risk register

Scoring: **Likelihood** and **Impact** as High/Medium/Low; **Rating** as the combination. Owner and treatment for each.

### 4.1 Blocking risks

| ID | Risk | L | I | Rating | Source |
|---|---|---|---|---|---|
| **R1** | **Unlicensed copyrighted material in the training set.** `real_studio` (494 clips) downloaded from YouTube via `yt-dlp`; URL lists committed. Exposure on platform ToS, third-party copyright, and speaker consent. **Structural** — removing it regresses studio FP from 12% toward 80%. | H | H | 🔴 **Critical** | VG-DOC-003 §I |
| **R2** | **Non-commercial datasets used in a commercial product.** WaveFake and XTTS (Coqui Public Model Licence) are likely non-commercial; both are baked into shipped weights. | H | H | 🔴 **Critical** | VG-DOC-003 §B, §G |

**Treatment for R1/R2 — legal review before enterprise sale.** If the position is untenable, remediation is re-sourcing and retraining, not redesign: Common Voice v26 (CC0, 27,929 Nigerian-language clips) is already held and would cover part of a replacement `real_studio`. **The cost rises the longer current weights remain in production**, because more customers come to rely on measured behaviour that a retrain would change.

### 4.2 High risks

| ID | Risk | L | I | Rating | Source |
|---|---|---|---|---|---|
| **R3** | **Deployed screener has no traceable provenance.** `v9h`'s `lcnn.pt` matches no retained source artifact; it makes ~86% of production decisions. | H | H | 🟠 High | VG-DOC-004 §9.1 |
| **R4** | **Silent sub-model failure can reach production.** No control validates that each sub-model contributes signal — the exact path by which AASIST reached production at AUC 0.5258. | M | H | 🟠 High | VG-DOC-004 §9.4 |
| **R5** | **Published metrics were not measured on the deployed bundle.** EER 2.43% / catch 99.2% / studio FP 12.0% are V9 figures; `v9h` differs precisely in the sub-model later found inert. The model card presents them under a `v9h` heading. | H | H | 🟠 High | VG-DOC-001 §16.2 |
| **R6** | **A published performance figure is wrong.** V9 val EER 13.25% was computed with an inverted softmax; the correct value is 16.19%. It propagated into multiple documents. | H | M | 🟠 High | VG-DOC-001 §16.1 |
| **R7** | **Adversarial evasion is not mitigated.** Ensemble EER ~0.48 under PGD; adversarial training had no effect on PGD. | M | H | 🟠 High | VG-DOC-001 §6 |
| **R8** | **Unreconciled capability discrepancy.** An internal finding that V9 misses ~half of TTS/multilingual fakes conflicts with the 99.2% catch figure. Unresolved. | M | H | 🟠 High | VG-DOC-001 §16.3 |
| **R9** | **Bias-audit claims are broader than the evidence.** Parity PASS rests on 5 of 7 languages; Pidgin tested with Nigerian-English TTS; Hausa train/test fakes from a single generation run; 25 of 50 Hausa test files corrupt. | H | M | 🟠 High | VG-DOC-003 §J |
| **R10** | **No independent benchmark.** ASVspoof 2021 harness built, never run. ASVspoof 2019 is a *training* corpus — a sibling release. | H | M | 🟠 High | VG-DOC-001 §16.4 |

### 4.3 Medium risks

| ID | Risk | L | I | Rating | Source |
|---|---|---|---|---|---|
| R11 | **Customer treats an advisory verdict as determinative**, producing an adverse outcome for an individual with no human review. | M | H | 🟡 Medium | §3.2 |
| R12 | **`auth_keys.json` loss revokes every client.** Single-file credential store. Mitigated by nightly encrypted backup — **only if `VOICEGUARD_BACKUP_KEY` is actually set**. | L | H | 🟡 Medium | Deployment design §8 |
| R13 | **Single-droplet architecture** — single point of failure, vertical scaling only. | M | M | 🟡 Medium | Deployment design §2 |
| R14 | **No named human approver** for model promotion. Actors are process labels. | H | L | 🟡 Medium | VG-DOC-004 §9.5 |
| R15 | **No training-data erasure mechanism.** A valid erasure request could not be honoured. | L | H | 🟡 Medium | VG-DOC-003 §6 |
| R16 | **Screener performance may not reproduce.** 8.18% EER is a best-of-50 from a run oscillating between 8% and 57%. | M | M | 🟡 Medium | VG-DOC-001 §16.6 |
| R17 | **Third-party model dependency.** `facebook/wav2vec2-base` is baked in at build; upstream licence and availability are not tracked. | L | M | 🟡 Medium | Dockerfile |
| R18 | **TTS provider ToS exposure** on generated training audio (ElevenLabs, edge-tts/Azure, gTTS, noiz.ai). | M | M | 🟡 Medium | VG-DOC-003 §8.3 |
| R19 | **Source checkpoints not integrity-managed.** `models/` has no manifest; a substitution would propagate into the next bundle and be hashed as legitimate. | L | M | 🟡 Medium | VG-DOC-004 §9.6 |
| R20 | **No documented retention policy.** Runtime behaviour is correct but unwritten; `jobs.db` grows without rotation. | H | L | 🟡 Medium | §1.2 |

### 4.4 Risk summary

| Rating | Count | IDs |
|---|---|---|
| 🔴 Critical | 2 | R1, R2 |
| 🟠 High | 8 | R3–R10 |
| 🟡 Medium | 10 | R11–R20 |

**Concentration:** 6 of the 10 highest risks (R1, R2, R5, R6, R8, R9) are **documentation, provenance, or measurement** risks rather than engineering defects. The system largely works; the record of *how* and *how well* is what needs strengthening.

---

## 5. Existing controls — with evidence

These are real, verifiable controls. Each row cites where an auditor can see proof.

### 5.1 AI lifecycle and model governance

| Control | Implementation | Evidence | ISO 42001 area |
|---|---|---|---|
| Versioned model registry | `bundle_registry.py` — register/promote/rollback | `model_store/registry.jsonl` | AI system life cycle |
| **Tamper-evident promotion log** | `ACTIVE.json`, hash-chained (`prev_sha` → `entry_sha`), append-only, atomic write | 4 entries, chain intact | Life cycle · Logging |
| **Per-file integrity verification** | SHA-256 per file; recomputed on `verify` and `pull` | `verify v9h` → **OK** (2026-07-22) | Life cycle |
| **Rollback capability** | One command; **exercised in production** — `v9fixed` promoted and rolled back in 3h 26m | `ACTIVE.json` seq 2→3 | Life cycle |
| **Fail-closed deployment** | `startup_check()` refuses to boot a bundle that cannot classify its fixture | `api.py` lifespan | Life cycle |
| **Deterministic inference** | Same input → same score, pinned by regression test | `tests/test_golden.py` → 1 passed | Life cycle |
| Golden regression baseline | 4 clips, scores + verdicts pinned at 1e-3 | `tests/golden_manifest.json` | Life cycle |
| Drift monitoring | Living test set re-scored; EER/catch drift detection; retrain trigger | `drift_monitor_3.py`, `GET /drift` | Life cycle |
| Model card | Architecture, use, data, evaluation, limitations, ethics | `docs/MODEL_CARD.md` | Information for interested parties |
| **Development history** | Full lineage V1→v9h incl. failures and rejections | VG-DOC-001 | Life cycle |

### 5.2 Data governance

| Control | Implementation | Evidence | Area |
|---|---|---|---|
| **Customer audio deleted after processing** | `worker.py` unlinks in a `finally` block — survives exceptions | `worker.py` | Data for AI systems |
| **No training on customer data** | No pipeline ingests production audio | VG-DOC-003 §7 | Data |
| Dataset inventory | 12 datasets, verified counts, provenance | VG-DOC-003 | Data |
| Manifest composition control | Per-bucket caps; warns at >2:1 class imbalance | `build_v9_manifest.py` | Data |
| Bias audit | 599 samples, 7 languages, formal report | VG-DOC-003 §J | Impact assessment |
| Upload size limit | 25 MiB at proxy **and** application | `deploy/Caddyfile`, `api.py` | Data |
| Weights excluded from source control | `.gitignore` / `.dockerignore` | Verified by test | Data |

### 5.3 Information security

| Control | Implementation | Evidence | ISO 27001 area |
|---|---|---|---|
| Authentication | Bearer keys, SHA-256 hashed; plaintext shown once | `auth.py` | Access control |
| Authorization scoping | Job results readable only by the issuing key; 404 on mismatch (no existence leak) | `api.py` | Access control |
| Rate limiting / anomaly detection | Per-key limits, duplicate detection, burst flags, `Retry-After` | `request_protection.py` | Monitoring |
| Encryption in transit | TLS via Caddy internal CA, VPC-private | `deploy/Caddyfile` | Cryptography |
| Network isolation | VPC-private bind; Cloud Firewall restricts to the backend IP | `docker-compose.prod.yml` | Network security |
| **Secrets excluded from images** | `auth_keys.json`, `jobs.db`, `.env` excluded; enforced by test | `tests/test_docker_context.py` | Secure development |
| Encrypted backups | Fernet; SQLite online-backup API for consistency | `deploy/backup.py` + tests | Backup |
| Immutable deployment | SHA-tagged images from a registry; rollback by tag | CI pipeline | Change management |
| Separation of environments | Dev / handoff / production Compose files | Repo | Separation |
| Automated test gate | Fast + weights tiers gate every change | `.github/workflows/ci.yml` | Secure development |
| Dependency pinning | Fully pinned `requirements.txt` | Repo | Supply chain |

### 5.4 Transparency and explainability

| Control | Implementation | Evidence |
|---|---|---|
| Per-detection explainability | Grad-CAM heatmap, SHAP attribution, flagged segments, prosody notes, confidence | `detector.py`, `gradcam.py` |
| Audit identifier | `audit_id` per detection | `explain_signals.py` |
| Legal report template | Non-technical, with limitations table and chain-of-custody | `forensic_report.py` + Phase 6 template |
| Threshold rationale | Four use cases with cost-function reasoning | Phase 6 thresholds document |
| Graceful degradation | Explainability failure never breaks detection (nulls out) | `detector.py` |

---

## 6. Control gaps

| ID | Gap | Consequence | Priority |
|---|---|---|---|
| **G1** | **No per-sub-model health check** | Silent model failure reaches production (has happened) | **Critical** |
| **G2** | **No provenance for the deployed screener** | Cannot demonstrate how 86% of decisions are produced | **Critical** |
| G3 | No AI policy, no risk-treatment plan, no documented objectives | No management system to certify | High |
| G4 | No named accountable approver for promotions | Change control cannot be evidenced | High |
| G5 | No formal data-retention or deletion policy | Correct behaviour is undocumented | High |
| G6 | No independent benchmark result | Performance claims are entirely self-reported | High |
| G7 | Audit log not wired into the live `/detect` path | Detection events are not tamper-evidently logged | High |
| G8 | No supplier / third-party register | R17, R18 untracked | Medium |
| G9 | No incident-response procedure for model failure | Ad hoc response to the next AASIST-class event | Medium |
| G10 | No documented competence / training requirements | ISO 42001 expects defined roles | Medium |
| G11 | No customer complaint / redress channel | Cannot evidence appeal rights | Medium |
| G12 | `jobs.db` grows without rotation | Unbounded metadata retention | Medium |
| G13 | No penetration test | Phase 7 exit gate unmet | Medium |
| G14 | Legal explainability template not reviewed by a non-technical reader | Phase 6 exit gate unmet | Medium |
| G15 | No membership-inference assessment | Cannot state whether training subjects are recoverable | Low |

---

## 7. Policies to author

The concrete deliverables. Work through as a checklist; drafting notes point at the source material.

| # | Policy | Must cover | Draw on |
|---|---|---|---|
| 1 | **AI Management Policy** | Scope, objectives, roles, review cycle, continual improvement | §1, §2 |
| 2 | **AI Risk Management Procedure** | Identification, scoring, treatment, acceptance authority | §4 — adopt the register wholesale |
| 3 | **Data Governance & Provenance Policy** | Dataset approval before use, **licence verification as a gate**, provenance records, prohibited sources | VG-DOC-003 — **make R1 impossible to repeat** |
| 4 | **Data Retention & Deletion Policy** | Audio deleted post-processing; `jobs.db` retention period; backup retention; erasure requests | §1.2, G5, G12 |
| 5 | **Model Change Control Procedure** | Registration, **acceptance gates**, named approver, promotion, rollback, customer notification | VG-DOC-004 §8, G4 |
| 6 | **Model Validation & Acceptance Criteria** | Required tests before promotion — **including a per-sub-model health check** | G1, VG-DOC-001 §17 item 8 |
| 7 | **Acceptable Use & Customer Responsibility Statement** | Advisory not determinative; human oversight; prohibited uses; §3.4 conditions | §3 |
| 8 | **Transparency & Disclosure Standard** | What is published, how limitations are disclosed, how metrics are sourced | R5, R6 |
| 9 | **Incident Response — AI Failure** | Detection, rollback, customer notification, post-incident review | G9 |
| 10 | **Third-Party & Supply Chain Policy** | Model, dataset, TTS-provider and infrastructure dependencies | G8, R17, R18 |
| 11 | **Bias & Fairness Monitoring Procedure** | Audit cadence, coverage requirements, parity thresholds, **honest scope statements** | R9 |
| 12 | **Information Security Policy** | Access, cryptography, secrets, backup, network | §5.3 — largely already implemented |
| 13 | **Roles, Responsibilities & Competence** | Named owners; segregation of duties | G10 |
| 14 | **Complaint & Redress Procedure** | How an affected individual contests an outcome | G11 |

**Drafting guidance on two of these.**

**Policy 3 is the one that matters most.** R1 and R2 exist because dataset acquisition was a technical decision with no licence gate. The policy must require licence verification and provenance recording *before* data enters a manifest, with a named approver. That single control would have prevented both critical risks.

**Policy 6 is where the sharpest lesson lives.** The AASIST collapse passed every existing gate — checkpoint selection, hash verification, golden regression, startup smoke check — because all of them test the *ensemble*. Acceptance criteria must include per-sub-model assertions: standalone AUC above a floor, non-degenerate score spread. The probe that eventually caught it (`aasist_probe.py`) already exists; it simply needs to be a gate rather than a manual investigation.

---

## 8. Standards mapping

> **Confirm clause references against the standard texts before audit.** The mapping below is indicative and by control *area*, not certified.

### 8.1 ISO/IEC 42001 (AI management system)

| Annex A area | Coverage | Evidence | Status |
|---|---|---|---|
| Policies related to AI | ❌ None written | — | **G3** |
| Internal organisation / roles | ❌ Undefined | — | **G10** |
| Resources for AI systems | ✅ Strong | VG-DOC-003, VG-DOC-004 | Good |
| Impact assessment | 🟡 Partial | Bias audit; no full impact assessment | Partial |
| **AI system life cycle** | ✅ **Strong** | Registry, promotion chain, rollback, golden regression, VG-DOC-001 | **Best area** |
| Data for AI systems | 🟡 Partial | Inventory complete; **licensing unresolved** | R1, R2 |
| Information for interested parties | ✅ Good | Model card, explainability, legal template | Good |
| Use of AI systems | 🟡 Partial | Thresholds documented; acceptable-use policy missing | Policy 7 |
| Third parties and customers | ❌ No register | — | **G8** |

### 8.2 ISO/IEC 27001 (information security)

| Control area | Coverage | Evidence |
|---|---|---|
| Asset inventory | ✅ | VG-DOC-004 |
| Access control / authentication | ✅ | `auth.py`, scoped job reads |
| Cryptography | ✅ | TLS, Fernet backups, SHA-256 integrity |
| Secure development | ✅ | CI gates, secrets-exclusion test, pinned deps |
| Configuration & change management | ✅ | Immutable images, registry, rollback |
| Backup | ✅ | Encrypted, consistency-safe, tested |
| Logging & monitoring | 🟡 | Drift + healthchecks; **detection audit log not wired (G7)** |
| Network security | ✅ | VPC-private, Cloud Firewall |
| Supplier relationships | ❌ | **G8** |
| Incident management | ❌ | **G9** |
| Vulnerability management | 🟡 | Dependencies pinned; **no pen test (G13)** |

---

## 9. Consolidated actions

| # | Action | Owner | Priority | Addresses |
|---|---|---|---|---|
| 1 | Legal review of training-data licensing | Counsel | **Critical** | R1, R2 |
| 2 | Add per-sub-model health check to the promotion gate | ML/Eng | **Critical** | G1, R4 |
| 3 | Re-derive and document the deployed screener's provenance | ML | **Critical** | G2, R3 |
| 4 | Correct the 13.25% → 16.19% figure everywhere it appears | ML | High | R6 |
| 5 | Re-measure bias audit + held-out on `v9h`; restate all claims | ML | High | R5 |
| 6 | Reconcile the recall discrepancy under one protocol | ML | High | R8 |
| 7 | Run ASVspoof 2021 LA and publish the result | ML | High | R10, G6 |
| 8 | Author Policies 1–6 (§7) | GRC | High | G3, G5 |
| 9 | Require a named approver on promote/rollback | Eng | High | G4, R14 |
| 10 | Wire the audit log into the live `/detect` path | Eng | High | G7 |
| 11 | Restate bias-audit scope honestly in all materials | ML/GRC | High | R9 |
| 12 | Plan re-sourced `real_studio` from permissive data | ML | High | R1 |
| 13 | Author Policies 7–14 | GRC | Medium | G8–G11, G14 |
| 14 | Third-party / supplier register | GRC | Medium | G8, R17, R18 |
| 15 | Penetration test | Eng | Medium | G13 |
| 16 | `jobs.db` rotation | Eng | Medium | G12 |
| 17 | Verify `VOICEGUARD_BACKUP_KEY` is set in production | Eng | Medium | R12 |
| 18 | Non-technical review of the legal template | Owner | Medium | G14 |
| 19 | Membership-inference assessment | ML | Low | G15 |

---

## 10. Audit evidence index

Where to find proof of each claim.

| Claim | Evidence | How to verify |
|---|---|---|
| Deployed model identity | `model_store/v9h/bundle.json` | `bundle_registry.py verify v9h` |
| Model change history | `model_store/ACTIVE.json` | Inspect hash chain — 4 entries |
| Deterministic inference | `tests/golden_manifest.json` | `pytest tests/test_golden.py` |
| Customer audio deleted | `worker.py` `finally` block | Code review + `tests/test_worker.py` |
| Secrets not in images | `.dockerignore` | `pytest tests/test_docker_context.py` |
| Backups exclude audio | `deploy/backup.py` | `pytest tests/test_backup.py` |
| Auth cannot be bypassed | `api.py` dependencies | `pytest tests/test_auth.py`, `test_api.py` |
| Bias audit performed | Bias Audit Report + VG-DOC-003 §J | Read with §J's scope caveats |
| Full development history | VG-DOC-001 | Cross-check against the training notebook |
| Dataset provenance | VG-DOC-003 | Filesystem enumeration |
| Known limitations disclosed | `MODEL_CARD.md` §7, VG-DOC-001 §17 | — |

---

## 11. Document change log

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-07-22 | Michael Ologungbara | Baseline. Consolidated risk register (2 critical, 8 high, 10 medium) from VG-DOC-001/003/004. Existing controls catalogued with evidence pointers across AI lifecycle, data governance, security and transparency. 15 control gaps identified. 14 policies specified for authoring. Indicative ISO/IEC 42001 and 27001 mapping. |
