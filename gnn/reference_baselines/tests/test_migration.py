"""Portable entry points, isolated staging and frozen-graph compatibility."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reference_baselines import prepare_inputs
from reference_baselines.paths import ROOT, PACKAGE_DIR


class MigrationTests(unittest.TestCase):
    def test_launcher_and_appendix_work_outside_repo_without_env(self):
        import os
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['REFERENCE_BASE_PATH'] = '/nonexistent/old/checkout'
        commands = [
            [ROOT / 'gnn/run.py', 'reference_set_comparison.generate_embeddings', '--help'],
            [ROOT / 'extra/edge_attribute_gnn/train.py', '--help'],
        ]
        with tempfile.TemporaryDirectory() as cwd:
            for command in commands:
                result = subprocess.run([sys.executable, *map(str, command)], cwd=cwd,
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_connectivity_launcher_uses_own_config_and_output(self):
        from gnn.run import resolve
        module, args = resolve('connectivity', ['train', '--smoke', '--graph', 'soap'])
        self.assertEqual(module, 'reference_baselines.training.train_reference_gnn')
        self.assertEqual(Path(args[args.index('--config') + 1]),
                         PACKAGE_DIR / 'configs/connectivity_reference_gnn.toml')
        self.assertEqual(Path(args[args.index('--run-dir') + 1]),
                         ROOT / 'outputs/gnn/connectivity_reference_gnn/smoke')
        _, override = resolve('connectivity', ['train', '--config=custom.toml', '--run-dir=custom'])
        self.assertEqual(override, ['--config=custom.toml', '--run-dir=custom'])

    def test_launcher_preserves_stage_and_data_options(self):
        from gnn.run import resolve
        module, args = resolve('proxy', ['report', '--model', 'esen'])
        self.assertEqual(module, 'reference_baselines.reference_set_comparison.pipeline')
        self.assertEqual(args, ['report', '--model', 'esen'])
        _, args = resolve('space-group', ['capture'])
        self.assertNotIn('--data', args)
        _, args = resolve('space-group', ['communities'])
        self.assertEqual(Path(args[args.index('--data') + 1]), ROOT / 'data/eval/inputs/references')
        with self.assertRaises(ValueError):
            resolve('reference', ['unknown'])

    def test_code_inputs_and_outputs_have_separate_roots(self):
        from reference_baselines.paths import DATA_DIR, RESULTS_DIR, COMPARISON_DIR, COMPARISON_DATA_DIR
        self.assertEqual(PACKAGE_DIR, ROOT / 'gnn/reference_baselines')
        self.assertFalse((ROOT / 'utils/eval/reference_baselines').exists())
        self.assertTrue(DATA_DIR.is_relative_to(ROOT / 'data'))
        self.assertEqual(RESULTS_DIR, ROOT / 'outputs/gnn')
        self.assertTrue(COMPARISON_DIR.is_relative_to(RESULTS_DIR))
        self.assertTrue(COMPARISON_DATA_DIR.is_relative_to(ROOT / 'data'))

    def test_staging_is_repeatable_and_does_not_alias_writable_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'frozen data')
            embedding = root / 'embeddings/input'
            result = root / 'results/graph'
            with (patch.object(prepare_inputs, 'input_links', return_value=[(source, embedding), (source, result)]),
                  patch.object(prepare_inputs, 'RESULTS_DIR', root / 'results'),
                  patch.object(prepare_inputs, 'DOCS_DIR', root / 'reports')):
                self.assertEqual(prepare_inputs.prepare(), [])
                self.assertEqual(prepare_inputs.prepare(), [])
                self.assertTrue(embedding.is_symlink())
                self.assertFalse(result.is_symlink())
                result.write_bytes(b'new graph')
                self.assertEqual(source.read_bytes(), b'frozen data')
                with self.assertRaises(FileExistsError):
                    prepare_inputs.prepare()

    def test_staging_copy_mode_and_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'input')
            dest = root / 'dest'
            missing = root / 'missing'
            with (patch.object(prepare_inputs, 'input_links', return_value=[(source, dest), (missing, root / 'unused')]),
                  patch.object(prepare_inputs, 'DOCS_DIR', root / 'reports')):
                self.assertEqual(prepare_inputs.prepare(copy=True), [missing])
                self.assertFalse(dest.is_symlink())
                self.assertEqual(dest.read_bytes(), b'input')
                self.assertFalse((root / 'unused').exists())

    def test_frozen_graph_digest_survives_path_change_but_rejects_algorithm_change(self):
        from evaluation.inference.graphs import _source_hashes
        from reference_baselines.datasets import decomposition
        name = decomposition.__name__
        # Recorded in the existing notebook graph_build.json before migration.
        expected = '69c4bfb375cf6ed2a2a5aac9d7e96bb687c5f5468e564a0ff9cf2e224ee7c1db'
        self.assertEqual(_source_hashes()[name], expected)
        read_text = Path.read_text
        def changed(path, *args, **kwargs):
            text = read_text(path, *args, **kwargs)
            if path == Path(decomposition.__file__):
                return text.replace('return {value: row', 'return {value: row + 1')
            return text
        with patch.object(Path, 'read_text', changed):
            self.assertNotEqual(_source_hashes()[name], expected)

    def test_proxy_paths_are_local_and_notebook_cells_compile(self):
        from reference_baselines.reference_set_comparison.configuration import load_config, PATH_KEYS
        c = load_config(PACKAGE_DIR / 'reference_set_comparison/experiment.yml')
        for key in PATH_KEYS:
            self.assertTrue(Path(c[key]).is_relative_to(ROOT), (key, c[key]))
        self.assertTrue(Path(c['jepa_config']).is_file())
        notebook = json.loads((PACKAGE_DIR / 'reference_set_comparison/inspect_train.ipynb').read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell {i}', 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main()
