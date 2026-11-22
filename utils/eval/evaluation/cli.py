"""Explicit stages; no automatic downloads or job submission."""
import argparse
import json
from .common import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','graphs','structures','recover-structure-receipt','score','select','phases','smoke','predict','hulls','analyse','appendix','export'])
    p.add_argument('--config')
    p.add_argument('--model',choices=['megnet','chgnet'])
    p.add_argument('--max-items',type=int)
    p.add_argument('--retry-errors',action='store_true')
    a = p.parse_args()
    c = load_config(a.config)
    if a.max_items is not None and a.max_items < 1:
        p.error('--max-items must be positive')
    if a.stage in ('smoke','predict','hulls') and not a.model:
        p.error('This stage requires --model')
    if a.stage=='audit':
        from .inputs import audit
        print(json.dumps(audit(c),indent=2))
    elif a.stage=='graphs':
        from .inference.graphs import build_graphs
        record = build_graphs(c)
        print(json.dumps({key:record[key] for key in ('mode','reference_count','fit_rows','physical','cosine')},indent=2))
    elif a.stage=='structures':
        from .inputs import prepare_structures
        prepare_structures(c)
    elif a.stage=='recover-structure-receipt':
        from .inputs import recover_structure_receipt
        print(json.dumps(recover_structure_receipt(c),indent=2))
    elif a.stage=='score':
        from .scoring import generate_scores
        generate_scores(c)
    elif a.stage=='select':
        from .selection import export_selections
        print(export_selections(c))
    elif a.stage=='phases':
        from .proxies import prepare_phases
        prepare_phases(c)
    elif a.stage=='smoke':
        from .proxies import smoke
        smoke(c,a.model)
    elif a.stage=='predict':
        from .proxies import predict
        predict(c,a.model,a.max_items,a.retry_errors)
    elif a.stage=='hulls':
        from .proxies import calculate_hulls
        calculate_hulls(c,a.model)
    elif a.stage=='analyse':
        from .analysis import analyse
        analyse(c)
    elif a.stage=='appendix':
        from .reporting import historical_appendix
        historical_appendix()
    elif a.stage=='export':
        from .transfer import export_results
        export_results(c)


if __name__=='__main__':
    main()
