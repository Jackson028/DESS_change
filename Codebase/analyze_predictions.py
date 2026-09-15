import argparse
import csv
import json
import os
from collections import Counter, defaultdict


DATASETS = ("14res", "14lap", "15res", "16res")


def has_metric_rows(run_dir):
    test_path = os.path.join(run_dir, "test_test.csv")
    if not os.path.exists(test_path):
        return False
    try:
        with open(test_path, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f, delimiter=";")]
    except (OSError, csv.Error):
        return False
    return any(row.get("epoch") not in (None, "", "epoch") for row in rows)


def latest_run_dir(log_path, dataset):
    dataset_dir = os.path.join(log_path, dataset)
    if not os.path.isdir(dataset_dir):
        return None

    candidates = [
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]
    candidates = [path for path in candidates if has_metric_rows(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_best_epoch(run_dir, metric):
    test_path = os.path.join(run_dir, "test_test.csv")
    with open(test_path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f, delimiter=";")]

    rows = [row for row in rows if row.get(metric) not in (None, "", metric)]
    if not rows:
        raise ValueError(f"No metric rows found in {test_path}")
    best = max(rows, key=lambda row: float(row[metric]))
    return str(best["epoch"])


def prediction_path_for_run(run_dir, epoch):
    pred_path = os.path.join(run_dir, "predict", f"predicted_test_epoch_{epoch}.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    return pred_path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def entity_key(entity, include_entity_types=False):
    if include_entity_types:
        return entity["start"], entity["end"], entity["type"]
    return entity["start"], entity["end"]


def doc_triplets(doc, include_entity_types=False):
    entities = [entity_key(entity, include_entity_types) for entity in doc["entities"]]
    triplets = set()
    for sentiment in doc["sentiments"]:
        head = entities[sentiment["head"]]
        tail = entities[sentiment["tail"]]
        triplets.add((head, tail, sentiment["type"]))
    return entities, triplets


def index_by_pair(triplets):
    by_pair = defaultdict(set)
    for head, tail, senti_type in triplets:
        by_pair[(head, tail)].add(senti_type)
    return by_pair


def sentiment_types_in(triplets):
    return {senti_type for _, _, senti_type in triplets}


def classify_fp(fp, gold_pairs, gold_entities, include_entity_types):
    head, tail, pred_type = fp
    pair = (head, tail)
    reverse_pair = (tail, head)

    if pair in gold_pairs:
        return "wrong_polarity_same_pair"
    if reverse_pair in gold_pairs:
        return "reversed_pair"

    if include_entity_types:
        head_known = head in gold_entities
        tail_known = tail in gold_entities
    else:
        head_known = head in gold_entities
        tail_known = tail in gold_entities

    if head_known and tail_known:
        return "wrong_pair_known_entities"
    return "boundary_or_entity_error"


def classify_fn(fn, pred_pairs, pred_entities):
    head, tail, gold_type = fn
    pair = (head, tail)
    reverse_pair = (tail, head)

    if pair in pred_pairs:
        return "missed_by_wrong_polarity"
    if reverse_pair in pred_pairs:
        return "missed_by_reversed_pair"
    if head in pred_entities and tail in pred_entities:
        return "missed_relation_between_found_entities"
    return "missed_entity_or_boundary"


def strip_entity_type(entity):
    return entity[:2]


def span_distance(span, candidates):
    span = strip_entity_type(span)
    if not candidates:
        return None
    distances = []
    for candidate in candidates:
        candidate = strip_entity_type(candidate)
        distances.append(abs(span[0] - candidate[0]) + abs(span[1] - candidate[1]))
    return min(distances)


def boundary_bucket(triplet, reference_entities):
    head, tail, _ = triplet
    head_dist = span_distance(head, reference_entities)
    tail_dist = span_distance(tail, reference_entities)
    distances = [dist for dist in (head_dist, tail_dist) if dist is not None]
    if not distances:
        return "no_reference_entity"
    max_dist = max(distances)
    if max_dist == 0:
        return "exact_entities_other_error"
    if max_dist <= 1:
        return "boundary_near_1"
    if max_dist <= 2:
        return "boundary_near_2"
    return "boundary_far"


def safe_div(num, den):
    return num / den * 100 if den else 0.0


def f1_from_pr(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def analyze_documents(gold_docs, pred_docs, include_entity_types=False, max_examples=0):
    if len(gold_docs) != len(pred_docs):
        raise ValueError(
            f"Gold/prediction length mismatch: {len(gold_docs)} vs {len(pred_docs)}"
        )

    totals = Counter()
    fp_categories = Counter()
    fn_categories = Counter()
    fp_boundary_buckets = Counter()
    fn_boundary_buckets = Counter()
    per_type = defaultdict(Counter)
    polarity_confusion = Counter()
    examples = []

    for doc_id, (gold_doc, pred_doc) in enumerate(zip(gold_docs, pred_docs)):
        gold_entities, gold_triplets = doc_triplets(gold_doc, include_entity_types)
        pred_entities, pred_triplets = doc_triplets(pred_doc, include_entity_types)

        gold_entities = set(gold_entities)
        pred_entities = set(pred_entities)
        gold_pairs = index_by_pair(gold_triplets)
        pred_pairs = index_by_pair(pred_triplets)

        tp = gold_triplets & pred_triplets
        fp = pred_triplets - gold_triplets
        fn = gold_triplets - pred_triplets

        totals["tp"] += len(tp)
        totals["fp"] += len(fp)
        totals["fn"] += len(fn)
        totals["gold"] += len(gold_triplets)
        totals["pred"] += len(pred_triplets)

        for triplet in tp:
            per_type[triplet[2]]["tp"] += 1
        for triplet in fp:
            category = classify_fp(triplet, gold_pairs, gold_entities, include_entity_types)
            fp_categories[category] += 1
            if category == "boundary_or_entity_error":
                fp_boundary_buckets[boundary_bucket(triplet, gold_entities)] += 1
            per_type[triplet[2]]["fp"] += 1
            head, tail, pred_type = triplet
            gold_types = gold_pairs.get((head, tail))
            if gold_types:
                for gold_type in gold_types:
                    polarity_confusion[(gold_type, pred_type)] += 1
        for triplet in fn:
            category = classify_fn(triplet, pred_pairs, pred_entities)
            fn_categories[category] += 1
            if category == "missed_entity_or_boundary":
                fn_boundary_buckets[boundary_bucket(triplet, pred_entities)] += 1
            per_type[triplet[2]]["fn"] += 1

        if max_examples and len(examples) < max_examples:
            for triplet in fp:
                examples.append(
                    {
                        "doc_id": doc_id,
                        "kind": "FP",
                        "category": classify_fp(
                            triplet, gold_pairs, gold_entities, include_entity_types
                        ),
                        "triplet": triplet_to_dict(triplet),
                        "gold": [triplet_to_dict(item) for item in sorted(gold_triplets, key=str)],
                        "tokens": gold_doc.get("tokens", pred_doc.get("tokens", [])),
                    }
                )
                if len(examples) >= max_examples:
                    break
        if max_examples and len(examples) < max_examples:
            for triplet in fn:
                examples.append(
                    {
                        "doc_id": doc_id,
                        "kind": "FN",
                        "category": classify_fn(triplet, pred_pairs, pred_entities),
                        "triplet": triplet_to_dict(triplet),
                        "pred": [triplet_to_dict(item) for item in sorted(pred_triplets, key=str)],
                        "tokens": gold_doc.get("tokens", pred_doc.get("tokens", [])),
                    }
                )
                if len(examples) >= max_examples:
                    break

    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    f1 = f1_from_pr(precision, recall)

    return {
        "totals": totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_categories": fp_categories,
        "fn_categories": fn_categories,
        "fp_boundary_buckets": fp_boundary_buckets,
        "fn_boundary_buckets": fn_boundary_buckets,
        "per_type": per_type,
        "polarity_confusion": polarity_confusion,
        "examples": examples,
    }


def triplet_to_dict(triplet):
    head, tail, senti_type = triplet
    return {"head": list(head), "tail": list(tail), "type": senti_type}


def print_counter(title, counter):
    print(title)
    if not counter:
        print("  -")
        return
    total = sum(counter.values())
    for key, value in counter.most_common():
        pct = safe_div(value, total)
        print(f"  {key}: {value} ({pct:.2f}%)")


def print_analysis(dataset, pred_path, result):
    totals = result["totals"]
    print(f"===== {dataset} =====")
    print(f"Prediction: {pred_path}")
    print(
        f"TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} "
        f"Gold={totals['gold']} Pred={totals['pred']}"
    )
    print(
        f"P={result['precision']:.4f} R={result['recall']:.4f} "
        f"F1={result['f1']:.4f}"
    )
    print_counter("FP categories:", result["fp_categories"])
    print_counter("FP boundary/entity distance:", result["fp_boundary_buckets"])
    print_counter("FN categories:", result["fn_categories"])
    print_counter("FN boundary/entity distance:", result["fn_boundary_buckets"])

    print("Per sentiment type:")
    if not result["per_type"]:
        print("  -")
    for senti_type in sorted(result["per_type"]):
        item = result["per_type"][senti_type]
        p = safe_div(item["tp"], item["tp"] + item["fp"])
        r = safe_div(item["tp"], item["tp"] + item["fn"])
        f1 = f1_from_pr(p, r)
        print(
            f"  {senti_type}: TP={item['tp']} FP={item['fp']} FN={item['fn']} "
            f"P={p:.2f} R={r:.2f} F1={f1:.2f}"
        )

    if result["polarity_confusion"]:
        print("Wrong-polarity confusion on exact pairs:")
        for (gold_type, pred_type), value in result["polarity_confusion"].most_common():
            print(f"  gold={gold_type} pred={pred_type}: {value}")
    print("")


def trim_like_reader(docs, enabled=True):
    if not enabled:
        return docs, 0
    remainder = len(docs) % 16
    if remainder == 0:
        return docs, 0
    return docs[:-remainder], remainder


def analyze_one(
    dataset,
    pred_path,
    gold_path,
    include_entity_types=False,
    max_examples=0,
    reader_trim=True,
):
    gold_docs = load_json(gold_path)
    gold_docs, removed = trim_like_reader(gold_docs, reader_trim)
    if removed:
        print(
            f"[INFO] {dataset}: trimmed {removed} gold samples to match InputReader "
            f"multiple-of-16 behavior."
        )
    pred_docs = load_json(pred_path)
    return analyze_documents(gold_docs, pred_docs, include_entity_types, max_examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, help="Dataset name for single-file mode")
    parser.add_argument("--pred", help="Prediction JSON path for single-file mode")
    parser.add_argument("--gold", help="Gold JSON path for single-file mode")
    parser.add_argument("--log_path", help="Analyze best-epoch predictions under a log root")
    parser.add_argument("--data_path", default="data", help="Dataset root path")
    parser.add_argument("--metric", default="senti_f1_micro", help="Metric used to select best epoch")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), help="Datasets for log mode")
    parser.add_argument("--include_entity_types", action="store_true", help="Use stricter NEC matching")
    parser.add_argument("--examples", type=int, default=0, help="Print a few FP/FN examples as JSON")
    parser.add_argument(
        "--no_reader_trim",
        action="store_true",
        help="Do not trim gold data to the multiple-of-16 length used by InputReader",
    )
    args = parser.parse_args()

    jobs = []
    if args.log_path:
        for dataset in args.datasets:
            run_dir = latest_run_dir(args.log_path, dataset)
            if run_dir is None:
                print(f"===== {dataset} =====")
                print(f"No valid run found under {args.log_path}")
                print("")
                continue
            epoch = read_best_epoch(run_dir, args.metric)
            pred_path = prediction_path_for_run(run_dir, epoch)
            gold_path = os.path.join(args.data_path, dataset, "test_dep_triple_polarity_result.json")
            jobs.append((dataset, pred_path, gold_path))
    else:
        if not (args.dataset and args.pred):
            parser.error("Use either --log_path or both --dataset and --pred")
        gold_path = args.gold or os.path.join(
            args.data_path, args.dataset, "test_dep_triple_polarity_result.json"
        )
        jobs.append((args.dataset, args.pred, gold_path))

    for dataset, pred_path, gold_path in jobs:
        result = analyze_one(
            dataset,
            pred_path,
            gold_path,
            include_entity_types=args.include_entity_types,
            max_examples=args.examples,
            reader_trim=not args.no_reader_trim,
        )
        print_analysis(dataset, pred_path, result)
        if args.examples:
            print("Examples:")
            print(json.dumps(result["examples"], ensure_ascii=False, indent=2))
            print("")


if __name__ == "__main__":
    main()
