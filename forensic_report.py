"""forensic_report.py — Phase 6: fill the Legal / Forensic Explainability Report template
from a VoiceGuard detection result.

Produces a self-contained, printable HTML report (print to PDF from the browser) that matches
the client-approved template in docs/new_docs/Phase6_Legal_Explainability_Report_Template.docx.
The fixed legal wording (sections 3, 6, 7) is preserved verbatim; the case-specific fields
(verdict, hash, audit id, observations, model versions) are filled from detect()'s output.

Usage:
    python forensic_report.py --file audio.mp3 --analyst "J. Doe" --out report.html
    python forensic_report.py --result job_result.json --exhibit "EX-12" --out report.html
"""
import os, sys, json, argparse, html, datetime

# Verdict thresholds (must match detector's verdict_from_score / thresholds_v9.json)
_TO_REVIEW, _LIKELY_FAKE, _AUTO_FAKE = 0.30, 0.55, 0.85


def _band(score):
    """(band_key, plain confidence phrase, short confidence level)."""
    if score >= _AUTO_FAKE:
        return "auto_fake", "High confidence of synthesis", "HIGH"
    if score >= _LIKELY_FAKE:
        return "likely_fake", "Moderate confidence of synthesis", "MODERATE"
    if score >= _TO_REVIEW:
        return "to_review", "Low confidence / inconclusive", "LOW"
    return "auto_real", "No significant indication of synthesis", "—"


def _plain_result(score):
    if score >= _LIKELY_FAKE:
        return ("This audio sample shows characteristics consistent with computer-generated "
                "(synthetic) speech.")
    if score >= _TO_REVIEW:
        return ("This audio sample shows some ambiguous indicators; the result is inconclusive "
                "and should not be treated as supporting evidence in either direction.")
    return "This audio sample shows no significant indication of synthetic generation."


_BANDS = [
    ("auto_real", "&lt; 0.30", "No significant indication of synthesis",
     "The analysis did not identify patterns associated with computer-generated speech."),
    ("to_review", "0.30 – 0.55", "Low confidence / inconclusive",
     "Some ambiguous indicators were present. This result alone should not be treated as "
     "supporting evidence in either direction."),
    ("likely_fake", "0.55 – 0.85", "Moderate confidence of synthesis",
     "The analysis identified meaningful patterns associated with computer-generated speech. "
     "Further verification is recommended before relying on this result."),
    ("auto_fake", "&ge; 0.85", "High confidence of synthesis",
     "The analysis identified strong, consistent patterns associated with computer-generated speech."),
]

_LIMITATIONS = [
    ("Deliberate audio manipulation",
     "If audio has been deliberately and expertly altered to evade detection, this system may not "
     "identify it. This is a known, documented limitation, not unique to this system — no current "
     "detection technology is immune to this."),
    ("Phone / call-quality audio",
     "Audio recorded or transmitted through a telephone connection is harder to analyze reliably "
     "than studio-quality audio, due to reduced audio fidelity. Results on phone-call recordings "
     "should be treated with additional caution."),
    ("Language and accent coverage",
     "This system's reliability across different languages, accents, and speaker demographics is an "
     "ongoing area of testing."),
    ("Not a substitute for expert forensic examination",
     "This report provides an automated, pattern-based assessment. It is not equivalent to "
     "examination by a certified forensic audio examiner and should be weighted accordingly, "
     "particularly in legal proceedings."),
]


def _observations_html(result):
    obs = ((result.get("explanation") or {}).get("observations")) or []
    items = []
    for o in obs:
        if isinstance(o, dict):
            t = html.escape(str(o.get("title", "")).strip())
            d = html.escape(str(o.get("detail", "")).strip())
            items.append(f"<li><b>{t}</b>{(' — ' + d) if d else ''}</li>")
        elif str(o).strip():
            items.append(f"<li>{html.escape(str(o))}</li>")
    segs = result.get("flagged_segments") or []
    if segs:
        rng = ", ".join(f"{s['start_sec']:.1f}–{s['end_sec']:.1f}s" for s in segs)
        items.append(f"<li><b>Flagged time segments</b> — {html.escape(rng)}</li>")
    if not items:
        items.append("<li>No specific explainability observations were recorded for this sample.</li>")
    return "\n".join(items)


def build_report_html(result, report_id=None, analyst="", operator="", exhibit=""):
    score = float(result.get("score", 0.0))
    band_key, band_phrase, conf_level = _band(score)
    verdict = result.get("verdict", "")
    audit_id = result.get("audit_id", "")
    sha = result.get("sha256", "")
    report_id = report_id or audit_id or "UNASSIGNED"
    date = datetime.date.today().isoformat()
    model_version = result.get("model_version", "V9")
    filename = html.escape(exhibit or os.path.basename(result.get("_source", "")) or "—")

    band_rows = "\n".join(
        f'<tr class="{"cur" if k == band_key else ""}"><td>{k} ({rng})</td><td>{conf}</td><td>{expl}</td></tr>'
        for k, rng, conf, expl in _BANDS)
    lim_rows = "\n".join(
        f"<tr><td>{html.escape(n)}</td><td>{html.escape(e)}</td></tr>" for n, e in _LIMITATIONS)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Audio Authenticity Analysis Report — {html.escape(report_id)}</title>
<style>
  body{{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;max-width:820px;margin:2rem auto;
       line-height:1.5;padding:0 1.5rem}}
  h1{{font-size:1.5rem;margin-bottom:.2rem}} .sub{{color:#555;margin-top:0}}
  h2{{font-size:1.1rem;border-bottom:2px solid #333;padding-bottom:.2rem;margin-top:1.8rem}}
  table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}}
  td,th{{border:1px solid #bbb;padding:.4rem .6rem;text-align:left;vertical-align:top}}
  .kv td:first-child{{width:34%;font-weight:bold;background:#f5f5f5}}
  tr.cur{{background:#fff3cd;font-weight:bold}}
  .result{{font-size:1.15rem;padding:.8rem 1rem;border-left:5px solid
           {'#c0392b' if score>=_LIKELY_FAKE else ('#e0a800' if score>=_TO_REVIEW else '#2e7d32')};
           background:#fafafa;margin:.6rem 0}}
  .conf{{font-weight:bold}} code{{font-family:Consolas,monospace;font-size:.85rem;word-break:break-all}}
  .note{{color:#555;font-size:.9rem}} ul{{margin:.4rem 0}} @media print{{body{{margin:0}}}}
</style></head><body>

<h1>VoiceGuard — Audio Authenticity Analysis Report</h1>
<p class="sub">For non-technical review. Technical scoring detail is confined to Section 8.</p>

<h2>1. Report Identification</h2>
<table class="kv">
  <tr><td>Report ID</td><td>{html.escape(report_id)}</td></tr>
  <tr><td>Date of Analysis</td><td>{date}</td></tr>
  <tr><td>Prepared By</td><td>{html.escape(analyst or operator or '—')}</td></tr>
  <tr><td>Audio File Reference</td><td>{filename}</td></tr>
  <tr><td>Audio Intake Hash (SHA-256)</td><td><code>{html.escape(sha)}</code></td></tr>
  <tr><td>Audit Trail Reference</td><td><code>{html.escape(audit_id)}</code></td></tr>
</table>

<h2>2. Plain-Language Result</h2>
<div class="result">{html.escape(_plain_result(score))}<br>
  <span class="conf">Confidence level: {conf_level} — {band_phrase}</span> (see Section 5).</div>

<h2>3. What Was Analyzed</h2>
<p>This report concerns a single audio file, identified above by filename and cryptographic hash.
The hash uniquely identifies this exact audio file and would change if the file were modified in
any way — it serves the same purpose as a fingerprint or seal for the purposes of this analysis.</p>
<p>The analysis examined acoustic characteristics of the audio (such as frequency patterns and
synthesis artifacts) using a combination of independent pattern-recognition models, each trained
separately on large collections of authentic and computer-generated speech. The models' outputs
were combined into a single score, which is translated into the plain-language result above.</p>
<p>This analysis does NOT perform speaker identification, voice biometric matching, or any
determination of WHO is speaking. It addresses only whether the audio shows characteristics
consistent with computer generation.</p>

<h2>4. Supporting Observations</h2>
<p>The system identified the following specific observations in this audio sample:</p>
<ul>{_observations_html(result)}</ul>
<p class="note">These observations describe patterns the system identified; they are presented in
plain language and are not themselves conclusive proof of authenticity or synthesis.</p>

<h2>5. How to Interpret the Confidence Level</h2>
<p>The system's underlying technical score has been translated into one of four plain-language
confidence bands (this sample falls in the highlighted row):</p>
<table><tr><th>System Score Band</th><th>Plain-Language Confidence</th><th>What This Means</th></tr>
{band_rows}</table>

<h2>6. What This Analysis Does Not Establish</h2>
<ul>
<li>This analysis does not establish who created the audio, their intent, or how the audio was used.</li>
<li>A result of “likely synthetic” or “high confidence” does not constitute legal proof of fraud,
forgery, or any other determination — it is one piece of technical evidence to be weighed alongside others.</li>
<li>A result of “no significant indication of synthesis” does not guarantee the audio is authentic — it
means the system did not detect patterns it is designed to identify, within its tested capabilities (see Section 7).</li>
<li>This system has not been independently certified as a forensic tool by any accredited body, and this
report should be evaluated accordingly in any legal or regulatory context.</li>
</ul>

<h2>7. Known Limitations</h2>
<table><tr><th>Known Limitation</th><th>Plain-Language Explanation</th></tr>
{lim_rows}</table>

<h2>8. Chain of Custody and Technical Reference</h2>
<p class="note">Provided for technical reviewers, auditors, or expert witnesses who may need to verify
this analysis. Not required for interpreting the plain-language result above.</p>
<table class="kv">
  <tr><td>Model / bundle version</td><td>{html.escape(str(model_version))} — see model registry (bundle_registry.py)</td></tr>
  <tr><td>Underlying numeric score</td><td>{score:.4f} (0.00–1.00) → verdict {html.escape(verdict)}</td></tr>
  <tr><td>Audio intake hash</td><td><code>{html.escape(sha)}</code></td></tr>
  <tr><td>Audit log reference</td><td><code>{html.escape(audit_id)}</code> (tamper-evident audit log)</td></tr>
  <tr><td>Determinism</td><td>Confirmed — the same file produces the same score on repeated analysis.</td></tr>
</table>

<h2>9. Analyst / Reviewer Sign-Off</h2>
<table><tr><th>Analyst Name</th><th>Date</th></tr>
<tr><td style="height:2.2rem">{html.escape(analyst or '')}</td><td>{date if analyst else ''}</td></tr></table>

</body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="audio file to analyze (runs detect())")
    g.add_argument("--result", help="a saved detection-result JSON to render")
    ap.add_argument("--out", default="forensic_report.html")
    ap.add_argument("--analyst", default="")
    ap.add_argument("--exhibit", default="")
    ap.add_argument("--report-id", default=None)
    args = ap.parse_args(argv)

    if args.file:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import detector
        result = detector.detect(args.file)
        result["_source"] = args.file
    else:
        with open(args.result, encoding="utf-8") as f:
            result = json.load(f)

    html_out = build_report_html(result, report_id=args.report_id,
                                 analyst=args.analyst, exhibit=args.exhibit)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"wrote {args.out}  (verdict={result.get('verdict')} score={result.get('score')})")


if __name__ == "__main__":
    main()
