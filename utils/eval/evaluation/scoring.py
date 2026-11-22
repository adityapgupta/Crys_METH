"""Exact baseline retrieval and label-free inference with the original GNN class."""
from itertools import combinations
import numpy as np
import pandas as pd
from .common import sha, read_json, write_json, verify, input_files
from .inputs import audit, checkpoints


def exact_neighbours(queries, references, k, metric='euclidean', chunk=128):
    """Float64 exhaustive retrieval; distance ties use the smaller reference row."""
    r = np.asarray(references, dtype=np.float64)
    if metric not in ('euclidean', 'cosine') or not 0 < k <= len(r):
        raise ValueError('Invalid retrieval metric or neighbour count')
    if metric == 'cosine':
        norm = np.linalg.norm(r, axis=1)
        if np.any(norm == 0):
            raise ValueError('Zero reference embedding')
        r = r/norm[:, None]
    ids, distances = [], []
    for start in range(0, len(queries), chunk):
        q = np.asarray(queries[start:start+chunk], dtype=np.float64)
        if metric == 'cosine':
            norm = np.linalg.norm(q, axis=1)
            if np.any(norm == 0):
                raise ValueError('Zero candidate embedding')
            d = np.clip(1-(q/norm[:, None])@r.T, 0, 2)
        else:
            d = np.maximum((q*q).sum(1)[:,None]+(r*r).sum(1)[None,:]-2*q@r.T, 0)
        order = np.argsort(d, axis=1, kind='stable')[:, :k]
        ids.append(order)
        distances.append(np.take_along_axis(d, order, axis=1))
    return np.concatenate(ids), np.concatenate(distances)


def chemical_pools(reference_metadata, systems):
    groups = {s: np.asarray(ix, dtype=np.int64)
              for s, ix in reference_metadata.groupby('chemical_system', sort=True).groups.items()}
    pools = {}
    for system in systems:
        elements = system.split('-')
        subs = ['-'.join(sub) for n in range(1,len(elements)+1) for sub in combinations(elements,n)]
        parts = [groups[sub] for sub in subs if sub in groups]
        pools[system] = np.sort(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)
    return pools


def score_chemical(queries, references, systems, pools):
    result = np.full(len(queries), np.nan)
    for i, (query, system) in enumerate(zip(queries, systems)):
        refs = pools[system]
        if len(refs):
            result[i] = np.square(np.asarray(references[refs], dtype=np.float64)-query).sum(1).mean()
    return result


def score_knn(queries, references, counts, metric):
    ids, distances = exact_neighbours(queries, references, max(counts), metric)
    return {f'knn_{metric}_{k}': distances[:,:k].mean(1) for k in counts}, ids


def compositions(formulas):
    from pymatgen.core import Composition
    result = np.zeros((len(formulas),100), dtype=np.float32)
    for i, formula in enumerate(formulas):
        comp = Composition(formula)
        for element in comp.elements:
            if element.Z >= 100:
                raise ValueError('Original GNN uses atomic-number indices 1..99, leaving index 0 unused')
            result[i, element.Z] = comp.get_atomic_fraction(element)
    return result


def feature_item(row, query, refs, reference_vectors, rcomp, qcomp, chemical, graph):
    """Match PoolDataset.__getitem__, without loading decomposition/stability targets."""
    import torch
    nodes = np.asarray(reference_vectors[refs], dtype=np.float32)
    embedding = np.asarray(query, dtype=np.float32)
    distance = np.square(np.asarray(nodes,dtype=np.float64)-np.asarray(embedding,dtype=np.float64)).sum(1)
    cosine = np.clip(1-(nodes@embedding)/np.maximum(
        np.linalg.norm(nodes,axis=1)*max(float(np.linalg.norm(embedding)),1e-12),1e-12),0,2)
    comp = rcomp[refs]
    _, inverse, counts = np.unique(comp, axis=0, return_inverse=True, return_counts=True)
    prior = np.log(1.0/(len(counts)*counts[inverse])).astype(np.float32)
    sub = graph[refs][:,refs].tocoo()
    return {'row':int(row), 'reference_ids':refs,
            'x':torch.from_numpy(np.concatenate([nodes,comp],axis=1)),
            'query':torch.from_numpy(np.r_[embedding,qcomp].astype(np.float32)),
            'pair':torch.from_numpy(np.column_stack([np.log1p(distance),np.isin(refs,chemical).astype(np.float32),cosine]).astype(np.float32)),
            # scatter_reduce used by GAT requires int64 index tensors in PyTorch 2.6.
            'edge':torch.from_numpy(np.array([sub.row,sub.col],dtype=np.int64)),
            'weight':torch.from_numpy(sub.data.astype(np.float32)),
            'target':torch.zeros(len(refs)), 'has_target':False,
            'log_prior':torch.from_numpy(prior), 'distance':torch.from_numpy(distance.astype(np.float32))}


def validate_training(c):
    """Use original training records to reject changed splits, graphs, or model source."""
    from .common import ROOT
    for graph in ('physical','cosine'):
        protocol_path = c['training_protocols']/f'train_protocol_{graph}.json'
        if not protocol_path.is_file():
            continue
        record = read_json(protocol_path)
        for rel, expected in record['input_sha256'].items():
            name = rel.replace('\\','/').split('/')[-1]
            path = input_files(c)[name] if name in ('train_rows.npy','validation_rows.npy','reserved_rows.npy') else c['graphs']/name
            if name in ('train_rows.npy','validation_rows.npy','reserved_rows.npy',f'{graph}.npz'):
                if sha(path) != expected:
                    raise ValueError(f'Training input mismatch: {name}')
        for rel, expected in record.get('source_sha256',{}).items():
            if rel.replace('\\','/').endswith(('models/reference_gnn.py','models/reference.py')):
                if sha(ROOT/rel) != expected:
                    raise ValueError(f'Trained model source changed: {rel}')


def score_gnns(c, candidates, queries, references, reference_metadata, pools, cosine_ids):
    import torch
    from scipy import sparse
    from reference_baselines.models.reference_gnn import ReferenceGNN
    from reference_baselines.datasets.reference_graph import collate
    validate_training(c)
    rcomp = compositions(reference_metadata.pretty_formula)
    qcomp = compositions(candidates.formula)
    loaded, records = {}, []
    for kind, graph, seed, folder in checkpoints(c):
        state = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        done = read_json(folder/'complete.json') if (folder/'complete.json').is_file() else None
        if (state['kind'], state['graph']) != (kind, graph):
            raise ValueError(f'Checkpoint provenance mismatch: {folder}')
        if done is not None and (done['seed'] != seed or done['reserved_opened'] or done['best_epoch'] != state['epoch']):
            raise ValueError(f'Checkpoint/completion epoch mismatch: {folder}')
        model = ReferenceGNN(kind,state['hidden'],state['layers']).eval()
        model.load_state_dict(state['state_dict'],strict=True)
        key = f'{kind}_{graph}_seed{seed}'
        loaded[key] = model
        records.append({'method':key,'sha256':sha(folder/'best.pt'),'epoch':state['epoch'],'historical_completion_available':done is not None})
    result = {key:np.empty(len(candidates)) for key in loaded}
    for graph in ('physical','cosine'):
        from .inference.graphs import load_graph
        adjacency = load_graph(c,graph)
        if adjacency.shape != (len(references),len(references)) or not np.isfinite(adjacency.data).all():
            raise ValueError('Invalid reference graph')
        for start in range(0,len(candidates),c['inference_batch_size']):
            stop = min(start+c['inference_batch_size'],len(candidates))
            items = []
            for i in range(start,stop):
                chemical = pools[candidates.iloc[i].chemical_system]
                refs = np.union1d(chemical,cosine_ids[i,:5]).astype(np.int64)
                items.append(feature_item(candidates.iloc[i].candidate_row,queries[i],refs,references,
                                          rcomp,qcomp[i],chemical,adjacency))
            batch = collate(items)
            with torch.inference_mode():
                for key, model in loaded.items():
                    if f'_{graph}_seed' in key:
                        result[key][start:stop] = model(batch)[2].numpy()
    return result, records


def generate_scores(c):
    report = audit(c)
    if (c['output']/'selection_complete.json').exists():
        frozen = verify(c['output'],'selection_complete.json')
        old = read_json(c['output']/'score_manifest.json')
        if old['input_sha256'] != report['input_sha256']:
            raise ValueError('Frozen scoring inputs changed; use a new output directory')
        if old['neighbours'] != c['neighbours'] or old['gnn_seeds'] != c['gnn_seeds']:
            raise ValueError('Frozen scoring settings changed')
        if frozen['fractions'] != c['fractions'] or frozen['seed'] != c['seed'] or frozen['batch_size'] != c['batch_size']:
            raise ValueError('Frozen selection configuration changed')
        return pd.read_csv(c['output']/'scores.csv',keep_default_na=False)
    if not report['ready_for_gnn']:
        raise FileNotFoundError('GNN scoring requires original artifacts:\n'+'\n'.join(report['blockers']))
    verify(c['output'],'structures_complete.json')
    candidates = pd.read_csv(c['output']/'candidates.csv',keep_default_na=False)
    reference_metadata = pd.read_csv(input_files(c)['metadata.csv'],keep_default_na=False)
    references = np.load(input_files(c)['embeddings.npy'],mmap_mode='r',allow_pickle=False)
    queries = np.load(input_files(c)['candidate_embeddings.npy'],mmap_mode='r',allow_pickle=False)[candidates.candidate_row]
    pools = chemical_pools(reference_metadata,candidates.chemical_system.unique())
    result = {'chemical':score_chemical(queries,references,candidates.chemical_system,pools)}
    for metric in ('euclidean','cosine'):
        values, ids = score_knn(queries,references,c['neighbours'],metric)
        result.update(values)
        if metric == 'cosine':
            cosine_ids = ids
    learned, records = score_gnns(c,candidates,queries,references,reference_metadata,pools,cosine_ids)
    result.update(learned)
    # A fixed random score yields nested random-candidate selections per batch.
    random_by_row = np.random.default_rng(c['seed']).random(len(np.load(input_files(c)['candidate_embeddings.npy'],mmap_mode='r')))
    result['random_candidates'] = random_by_row[candidates.candidate_row]
    table = candidates.copy()
    for name, values in result.items():
        table[name] = values
    table.to_csv(c['output']/'scores.csv',index=False)
    write_json(c['output']/'score_manifest.json',{'methods':list(result),'checkpoints':records,
               'neighbours':c['neighbours'],'gnn_seeds':c['gnn_seeds'],
               'input_sha256':report['input_sha256'],'graphs':{g:sha(c['graphs']/f'{g}.npz') for g in ('physical','cosine')},
               'note':'All GNNs use chemical union cosine-5 candidate pools, regardless of graph topology.'})
    return table
