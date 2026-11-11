from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

BASE = Path(__file__).resolve().parent
SOURCE = Path('/home/heheboiz/data/cir/workbench')
spec = importlib.util.spec_from_file_location('review_gate', Path('/home/heheboiz/data/cir/workbench/backend/parity_gate.py'))
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

test_file = SOURCE / 'tests/test_gate_contract_negatives.py'
tree = ast.parse(test_file.read_text())
names = {'_native_identity', '_native', '_forward', '_adapter'}
selected = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        selected.append(node)
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'TOLERANCES', 'CATEGORY'} for t in node.targets):
        selected.append(node)
ns = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(test_file), 'exec'), ns)

def evaluate(native, adapter, model='encoder', extra_categories=None):
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = {'shirt': (native, adapter)}
        pairs.update(extra_categories or {})
        for category, (n, a) in pairs.items():
            for side, obj in [('native', n), ('adapter', a)]:
                (root / f'g1_g2.{model}.{category}.{side}.json').write_text(json.dumps(obj))
        return gate.evaluate_model(model, root=root, categories=tuple(pairs)).to_dict()

native = ns['_native']()
adapter = ns['_adapter']()
results = {}

def keep(name, verdict):
    results[name] = {k:verdict[k] for k in ('status','passed','complete','missing','dimensions','failure_count')}

keep('complete_fixture_no_relation', evaluate(native, adapter))
keep('empty_adapter_control', evaluate(native, {**adapter,'adapter':{}}))
wrong_n, wrong_a = deepcopy(native), deepcopy(adapter)
wrong_n.update(model_id='model_A',checkpoint_id='checkpoint_A',protocol_id='protocol_A',query_order_sha256='query_A')
wrong_a.update(model_id='model_B',checkpoint_id='checkpoint_B',protocol_id='protocol_B',query_order_sha256='query_B')
keep('contradictory_identity', evaluate(wrong_n, wrong_a))
score = deepcopy(adapter)
score['adapter']['scoring_ranking'] = {'score_max_abs_delta_vs_dot':0.1,'score_max_abs_declared':1.0,'score_within_tolerance':True}
keep('local_tolerance_overrides_global_and_no_rank_evidence', evaluate(native, score))
thin = deepcopy(adapter)
thin['adapter']['scoring_ranking'] = {'score_max_abs_delta_vs_dot':0.0,'score_max_abs_declared':1e-6}
keep('score_only_no_rank_evidence', evaluate(native, thin))
keep('other_categories_empty', evaluate(native, adapter, extra_categories={
    'dress':(native,{**adapter,'adapter':{}}),
    'toptee':(native,{**adapter,'adapter':{}}),
}))
existing = gate.evaluate_all(root=Path('/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn'))
results['historical_verdict'] = {model:{k:v[k] for k in ('status','failure_count','missing')} for model,v in existing['models'].items()}
print(json.dumps(results,indent=2))
