import argparse
import csv
import os


DATASETS = ("14res", "14lap", "15res", "16res")


def latest_run_dir(log_path, dataset):
    dataset_dir = os.path.join(log_path, dataset)
    if not os.path.isdir(dataset_dir):
        return None

    candidates = [
        os.path.join(dataset_dir, name)
        for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]
    candidates = [
        path for path in candidates if has_metric_rows(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def has_metric_rows(run_dir):
    test_path = os.path.join(run_dir, "test_test.csv")
    if not os.path.exists(test_path):
        return False
    try:
        with open(test_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
    except (OSError, csv.Error):
        return False
    return bool(rows)


def _read_metric_rows(path, metric):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    rows = [row for row in rows if row.get(metric) not in (None, "", metric)]
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def read_best(run_dir, metric):
    test_path = os.path.join(run_dir, "test_test.csv")
    rows = _read_metric_rows(test_path, metric)
    best = max(rows, key=lambda row: float(row[metric]))
    last = rows[-1]
    return {
        "best": float(best[metric]),
        "best_epoch": best.get("epoch", ""),
        "last": float(last[metric]),
        "last_epoch": last.get("epoch", ""),
        "run_dir": run_dir,
    }


def read_result(run_dir, metric, selection="test_best"):
    if selection == "test_best":
        item = read_best(run_dir, metric)
        item.update(
            {
                "test": item["best"],
                "test_epoch": item["best_epoch"],
                "selection_score": item["best"],
                "selection_epoch": item["best_epoch"],
            }
        )
        return item

    if selection != "dev_best":
        raise ValueError(f"Unknown selection mode: {selection}")

    dev_path = _find_first_existing(
        run_dir,
        ["test_dev.csv", "dev_test.csv", "dev_dev.csv"],
    )
    test_path = os.path.join(run_dir, "test_test.csv")
    if dev_path is None:
        raise FileNotFoundError(
            f"No dev metric file found in {run_dir}; expected test_dev.csv"
        )
    dev_rows = _read_metric_rows(dev_path, metric)
    test_rows = _read_metric_rows(test_path, metric)

    best_dev = max(dev_rows, key=lambda row: float(row[metric]))
    best_epoch = best_dev.get("epoch", "")
    test_by_epoch = {row.get("epoch", ""): row for row in test_rows}
    if best_epoch not in test_by_epoch:
        raise ValueError(
            f"No test row for dev-selected epoch {best_epoch} in {test_path}"
        )

    selected_test = test_by_epoch[best_epoch]
    last = test_rows[-1]
    return {
        "best": float(selected_test[metric]),
        "best_epoch": best_epoch,
        "test": float(selected_test[metric]),
        "test_epoch": best_epoch,
        "selection_score": float(best_dev[metric]),
        "selection_epoch": best_epoch,
        "last": float(last[metric]),
        "last_epoch": last.get("epoch", ""),
        "run_dir": run_dir,
    }


def _find_first_existing(run_dir, names):
    for name in names:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="log", help="Root log directory")
    parser.add_argument(
        "--metric", default="senti_f1_micro", help="Metric column in test_test.csv"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS), help="Datasets to summarize"
    )
    parser.add_argument(
        "--selection",
        default="test_best",
        choices=["test_best", "dev_best"],
        help="Select test-best rows or report the test row at the best dev epoch",
    )
    args = parser.parse_args()

    results = []
    for dataset in args.datasets:
        run_dir = latest_run_dir(args.log_path, dataset)
        if run_dir is None:
            results.append((dataset, None))
            continue
        results.append((dataset, read_result(run_dir, args.metric, args.selection)))

    values = [item["best"] for _, item in results if item is not None]
    avg = sum(values) / len(values) if values else None

    print(f"Metric: {args.metric}")
    print(f"Log path: {args.log_path}")
    print(f"Selection: {args.selection}")
    print("")
    if args.selection == "dev_best":
        print(
            f"{'Dataset':<8} {'Test@Dev':>10} {'Epoch':>8} {'DevBest':>10} "
            f"{'Last':>10} {'LastEp':>8} Run"
        )
    else:
        print(
            f"{'Dataset':<8} {'Best':>10} {'Epoch':>8} {'Last':>10} "
            f"{'LastEp':>8} Run"
        )
    print("-" * 90)
    for dataset, item in results:
        if item is None:
            if args.selection == "dev_best":
                print(
                    f"{dataset:<8} {'N/A':>10} {'':>8} {'N/A':>10} "
                    f"{'N/A':>10} {'':>8} -"
                )
            else:
                print(f"{dataset:<8} {'N/A':>10} {'':>8} {'N/A':>10} {'':>8} -")
            continue
        if args.selection == "dev_best":
            print(
                f"{dataset:<8} {item['test']:>10.4f} {item['test_epoch']:>8} "
                f"{item['selection_score']:>10.4f} {item['last']:>10.4f} "
                f"{item['last_epoch']:>8} {item['run_dir']}"
            )
        else:
            print(
                f"{dataset:<8} {item['best']:>10.4f} {item['best_epoch']:>8} "
                f"{item['last']:>10.4f} {item['last_epoch']:>8} {item['run_dir']}"
            )
    print("-" * 90)
    if avg is not None:
        print(f"{'Avg':<8} {avg:>10.4f}")


if __name__ == "__main__":
    main()
