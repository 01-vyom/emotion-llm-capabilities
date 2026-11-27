"""Write the preregistered Stage 3.5 surface-matched vignette corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from emotion_stage1.affect import EMOTION_CONCEPT_PAIRS
from emotion_stage3.io import write_environment_snapshots, write_jsonl


_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

TOPICS = (
    {
        "topic": "A reviewer checks repeated invoice rows before a scheduled handoff",
        "assignment": "a queue of repeated invoice rows",
        "material": "amounts, dates, vendor codes, and approval fields",
        "interruption": "a colleague asked whether the packet could be closed early",
        "anomaly": "one subtotal that did not match its supporting lines",
        "handoff": "the accounting review queue",
    },
    {
        "topic": "A support analyst classifies nearly identical account messages",
        "assignment": "a batch of nearly identical account messages",
        "material": "case notes, label definitions, prior examples, and timestamps",
        "interruption": "a status alert requested an immediate progress estimate",
        "anomaly": "one message whose wording fit two neighboring labels",
        "handoff": "the customer support intake queue",
    },
    {
        "topic": "An auditor searches a routine report for a rare inconsistency",
        "assignment": "a routine operational report",
        "material": "control totals, exception codes, footnotes, and source records",
        "interruption": "a manager asked when the report would be ready",
        "anomaly": "one exception code that conflicted with its source record",
        "handoff": "the audit evidence folder",
    },
    {
        "topic": "A programmer applies one small repair across many similar files",
        "assignment": "a repeated repair across many similar files",
        "material": "diffs, tests, file names, and issue references",
        "interruption": "a build notification reported an unrelated warning",
        "anomaly": "one file whose surrounding logic differed from the pattern",
        "handoff": "the code review branch",
    },
    {
        "topic": "A legal assistant compares recurring clauses against a reference",
        "assignment": "a set of recurring contract clauses",
        "material": "clause text, defined terms, page references, and annotations",
        "interruption": "a calendar reminder announced the next review meeting",
        "anomaly": "one clause that used an older defined term",
        "handoff": "the counsel review packet",
    },
    {
        "topic": "A data analyst normalizes dates and amounts in a spreadsheet",
        "assignment": "a spreadsheet of dates and monetary amounts",
        "material": "formats, formulas, source columns, and validation notes",
        "interruption": "a teammate asked for an early copy of the cleaned sheet",
        "anomaly": "one date whose regional format was ambiguous",
        "handoff": "the shared analysis workbook",
    },
    {
        "topic": "A laboratory coordinator reconciles repeated sample records",
        "assignment": "a ledger of repeated sample records",
        "material": "sample identifiers, collection times, storage slots, and initials",
        "interruption": "a technician asked whether the rack could be released",
        "anomaly": "one sample identifier that appeared in two storage slots",
        "handoff": "the laboratory tracking system",
    },
    {
        "topic": "A logistics planner verifies shipment entries before dispatch",
        "assignment": "a manifest of scheduled shipment entries",
        "material": "destinations, weights, seal numbers, and departure windows",
        "interruption": "a dispatcher requested the current completion estimate",
        "anomaly": "one seal number that differed from the loading record",
        "handoff": "the dispatch control desk",
    },
    {
        "topic": "A researcher screens repetitive citations for eligibility",
        "assignment": "a list of repetitive research citations",
        "material": "titles, abstracts, inclusion rules, and reviewer notes",
        "interruption": "a collaborator asked how many records remained",
        "anomaly": "one abstract whose population description was incomplete",
        "handoff": "the evidence screening database",
    },
    {
        "topic": "A compliance specialist checks recurring access requests",
        "assignment": "a queue of recurring access requests",
        "material": "roles, approvals, expiration dates, and policy references",
        "interruption": "an operations message asked for the next approved batch",
        "anomaly": "one request whose approval expired before its start date",
        "handoff": "the access administration queue",
    },
    {
        "topic": "An editor reviews repeated table entries before publication",
        "assignment": "a set of repeated table entries",
        "material": "captions, units, references, and source values",
        "interruption": "a production message asked whether layout could begin",
        "anomaly": "one unit label that disagreed with the source value",
        "handoff": "the publication production folder",
    },
    {
        "topic": "A quality inspector compares recurring assembly records",
        "assignment": "a sequence of recurring assembly records",
        "material": "measurements, lot numbers, tolerances, and signoff fields",
        "interruption": "a floor supervisor requested the next release decision",
        "anomaly": "one measurement that sat exactly on a tolerance boundary",
        "handoff": "the quality release system",
    },
)

NAMES = (
    "Avery",
    "Blair",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
)

ORDER_CONTRASTS = {
    "calm_vs_desperate": (
        "Planning preceded action, breathing preceded response, and review preceded closure.",
        "Action preceded planning, response preceded breathing, and closure preceded review.",
    ),
    "focused_vs_distracted": (
        "The target preceded the interruption, the checklist preceded the message, and attention preceded switching.",
        "The interruption preceded the target, the message preceded the checklist, and switching preceded attention.",
    ),
    "patient_vs_rushed": (
        "Confirmation preceded action, waiting preceded filing, and evidence preceded closure.",
        "Action preceded confirmation, filing preceded waiting, and closure preceded evidence.",
    ),
    "responsible_vs_careless": (
        "Verification preceded signoff, comparison preceded acceptance, and evidence preceded closure.",
        "Signoff preceded verification, acceptance preceded comparison, and closure preceded evidence.",
    ),
    "curious_vs_bored": (
        "Investigation preceded dismissal, questions preceded routine, and inspection preceded closure.",
        "Dismissal preceded investigation, routine preceded questions, and closure preceded inspection.",
    ),
}


def _bag(text: str) -> Counter[str]:
    return Counter(_WORD_RE.findall(text.lower()))


def build_story(
    topic: dict[str, str],
    pair_name: str,
    *,
    positive: bool,
    replicate: int,
) -> str:
    name = NAMES[replicate % len(NAMES)]
    batch_id = f"{(replicate * 7919 + 104729) % 1_000_000:06d}"
    contrast = ORDER_CONTRASTS[pair_name][0 if positive else 1]
    sentences = (
        f"At the beginning of the shift, {name} received {topic['assignment']}, "
        f"a routine job identified as batch {batch_id} and scheduled for one "
        "ordinary handoff later that day.",
        f"The workspace contained {topic['material']}, along with a fixed "
        "reference checklist, a blank exception log, and the same quiet set of "
        "tools used for every comparable batch.",
        f"During the first pass, {name} moved through the material in its given "
        "order, recorded each completed entry, and left unchanged any field that "
        "the source did not support.",
        f"Halfway through the queue, {topic['interruption']}; the request added "
        "no new evidence and did not change the stated acceptance rules for the "
        "assignment.",
        contrast,
        f"When {topic['anomaly']} appeared, {name} kept the same records, tools, "
        "and deadline in view while deciding the order of the remaining actions "
        "and documenting what happened.",
        f"After the last recurring item, {name} compared the completion count "
        "with the batch header, attached the exception note, and prepared the "
        "unchanged source material for transfer.",
        f"The finished packet entered {topic['handoff']} with its timestamps, "
        "references, and outcome preserved, allowing another reviewer to "
        "reconstruct the sequence without relying on memory or unstated context.",
    )
    return " ".join(sentences)


def build_rows(replicates: int) -> list[dict]:
    rows = []
    for pair_index, (positive_name, negative_name) in enumerate(
        EMOTION_CONCEPT_PAIRS
    ):
        pair_name = f"{positive_name}_vs_{negative_name}"
        for topic_index, topic in enumerate(TOPICS):
            for replicate in range(replicates):
                pair_id = (
                    f"stage35-pair-{pair_index}-topic-{topic_index}-rep-{replicate}"
                )
                pair_rows = []
                for pole, emotion, contrast_emotion, positive in (
                    ("positive", positive_name, negative_name, True),
                    ("negative", negative_name, positive_name, False),
                ):
                    text = build_story(
                        topic,
                        pair_name,
                        positive=positive,
                        replicate=replicate,
                    )
                    word_count = len(text.split())
                    if not 150 <= word_count <= 250:
                        raise ValueError(
                            f"Curated vignette has {word_count} words; expected 150-250"
                        )
                    pair_rows.append(
                        {
                            "vignette_id": len(rows) + len(pair_rows),
                            "emotion": emotion,
                            "topic": topic["topic"],
                            "replicate": replicate,
                            "pair_name": pair_name,
                            "pair_id": pair_id,
                            "pole": pole,
                            "contrast_emotion": contrast_emotion,
                            "strict_stimuli": True,
                            "curated_stimuli": True,
                            "text": text,
                            "word_count": word_count,
                            "names_target_concept": bool(
                                re.search(
                                    rf"\b{re.escape(emotion)}\b",
                                    text,
                                    flags=re.IGNORECASE,
                                )
                            ),
                        }
                    )
                if _bag(pair_rows[0]["text"]) != _bag(pair_rows[1]["text"]):
                    raise ValueError(f"Pair {pair_id} is not bag-of-words matched")
                rows.extend(pair_rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise SystemExit("--replicates must be positive")
    output = args.output_dir / "vignettes.jsonl"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.replicates)
    write_jsonl(output, rows)
    metadata = {
        "stage": 3.5,
        "artifact": "vignettes",
        "generator": "curated_surface_matched_order_contrast_v1",
        "replicates": args.replicates,
        "num_topics": len(TOPICS),
        "num_vignettes": len(rows),
        "num_matched_pairs": len(rows) // 2,
        "word_count_min": min(row["word_count"] for row in rows),
        "word_count_max": max(row["word_count"] for row in rows),
        "named_target_concept_count": sum(
            row["names_target_concept"] for row in rows
        ),
        "pair_bow_exact_rate": 1.0,
        "contrast_location": "after_token_50",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
