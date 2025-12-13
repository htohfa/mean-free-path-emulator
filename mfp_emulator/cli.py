"""
Command-line interface for mfp_emulator.
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="MFP Emulator - Neural network emulator for mean free path predictions"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new emulator')
    train_parser.add_argument('data', help='Path to training data file')
    train_parser.add_argument('-o', '--output', default='emulator.pt', help='Output model path')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden layer dimension')
    train_parser.add_argument('--n-blocks', type=int, default=6, help='Number of residual blocks')
    train_parser.add_argument('--energies', type=float, nargs='+', help='Energies to include (eV)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('model', help='Path to trained model')
    predict_parser.add_argument('--z', type=float, required=True, help='Redshift')
    predict_parser.add_argument('--z-re', type=float, required=True, help='Reionization redshift')
    predict_parser.add_argument('--gamma', type=float, required=True, help='Gamma_12')
    predict_parser.add_argument('--density', type=float, default=0.0, help='Density in sigma units')
    predict_parser.add_argument('--energy', type=float, default=13.6, help='Energy in eV')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests from config file')
    test_parser.add_argument('config', help='Path to test config file')
    test_parser.add_argument('model', help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_train(args)
    elif args.command == 'predict':
        run_predict(args)
    elif args.command == 'test':
        run_test(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_train(args):
    """Run training."""
    from .emulator import MFPEmulator
    
    print(f"Loading data from {args.data}")
    emulator = MFPEmulator(data_path=args.data)
    
    if args.energies:
        emulator.load_data(energies=args.energies)
    else:
        emulator.load_data()
    
    print("Preparing data splits...")
    emulator.prepare_data()
    
    print(f"Training (epochs={args.epochs}, hidden_dim={args.hidden_dim}, n_blocks={args.n_blocks})")
    emulator.train(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        verbose=True
    )
    
    print("Evaluating on test set...")
    emulator.evaluate()
    
    print(f"Saving model to {args.output}")
    emulator.save(args.output)
    print("Done!")


def run_predict(args):
    """Run single prediction."""
    from .emulator import MFPEmulator
    
    emulator = MFPEmulator()
    emulator.load(args.model)
    
    mfp = emulator.predict(
        z=args.z,
        z_re=args.z_re,
        gamma=args.gamma,
        density=args.density,
        energy_eV=args.energy
    )
    
    print(f"MFP prediction: {mfp:.4f} cMpc")
    print(f"  z={args.z}, z_re={args.z_re}, gamma_12={args.gamma}")
    print(f"  density={args.density} sigma, E={args.energy} eV")


def run_test(args):
    """Run tests from config file."""
    from .emulator import MFPEmulator
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Parse config file
    tests = parse_config(config_path)
    
    print(f"Loading model from {args.model}")
    emulator = MFPEmulator()
    emulator.load(args.model)
    
    print(f"\nRunning {len(tests)} tests...")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests):
        test_type = test.get('type', 'predict')
        name = test.get('name', f'test_{i+1}')
        
        try:
            if test_type == 'predict':
                result = run_predict_test(emulator, test)
            elif test_type == 'reionization':
                result = run_reionization_test(emulator, test)
            elif test_type == 'ndot':
                result = run_ndot_test(emulator, test)
            else:
                print(f"  [{name}] Unknown test type: {test_type}")
                failed += 1
                continue
            
            if result['success']:
                print(f"  [{name}] PASSED - {result['message']}")
                passed += 1
            else:
                print(f"  [{name}] FAILED - {result['message']}")
                failed += 1
                
        except Exception as e:
            print(f"  [{name}] ERROR - {str(e)}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)


def parse_config(path: Path) -> list:
    """
    Parse test config file.
    
    Format (one test per line):
    type=predict z=5.0 z_re=7.0 gamma=0.5 [density=0.0] [energy=13.6] [expected_mfp=XX]
    type=reionization model=tanh z_re=7.0 Delta_z=1.5 z=5.0 [expected_Q=XX]
    type=ndot z=5.0 history=late_start_late_end.txt
    """
    tests = []
    
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            test = {'name': f'line_{line_num}'}
            
            # Parse key=value pairs
            parts = line.split()
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Try to convert to number
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    test[key] = value
            
            if 'type' in test:
                tests.append(test)
    
    return tests


def run_predict_test(emulator, test: dict) -> dict:
    """Run a prediction test."""
    required = ['z', 'z_re', 'gamma']
    for key in required:
        if key not in test:
            return {'success': False, 'message': f'Missing required key: {key}'}
    
    mfp = emulator.predict(
        z=test['z'],
        z_re=test['z_re'],
        gamma=test['gamma'],
        density=test.get('density', 0.0),
        energy_eV=test.get('energy', 13.6)
    )
    
    message = f"MFP={mfp:.4f} cMpc"
    
    if 'expected_mfp' in test:
        expected = test['expected_mfp']
        rel_error = abs(mfp - expected) / expected
        if rel_error < 0.1:  # 10% tolerance
            return {'success': True, 'message': f"{message} (expected {expected:.4f}, error {rel_error*100:.1f}%)"}
        else:
            return {'success': False, 'message': f"{message} (expected {expected:.4f}, error {rel_error*100:.1f}%)"}
    
    # Just check it's reasonable
    if mfp > 0 and mfp < 1e6:
        return {'success': True, 'message': message}
    else:
        return {'success': False, 'message': f"Unreasonable MFP: {mfp}"}


def run_reionization_test(emulator, test: dict) -> dict:
    """Run a reionization history test."""
    from .reionization import tanh_reionization_model, load_reionization_history
    
    model = test.get('model', 'tanh')
    z = test.get('z', 5.0)
    
    if model == 'tanh':
        z_re = test.get('z_re', 7.0)
        Delta_z = test.get('Delta_z', 1.5)
        Q = tanh_reionization_model(z, z_re, Delta_z)
    else:
        history_file = test.get('history_file')
        if not history_file:
            return {'success': False, 'message': 'Missing history_file for file model'}
        history = load_reionization_history(path=history_file, model='file')
        Q = history.Q(z)
    
    message = f"Q(z={z})={Q:.4f}"
    
    if 'expected_Q' in test:
        expected = test['expected_Q']
        if abs(Q - expected) < 0.05:
            return {'success': True, 'message': message}
        else:
            return {'success': False, 'message': f"{message} (expected {expected:.4f})"}
    
    if 0 <= Q <= 1:
        return {'success': True, 'message': message}
    else:
        return {'success': False, 'message': f"Invalid Q: {Q}"}


def run_ndot_test(emulator, test: dict) -> dict:
    """Run an ndot calculation test."""
    from .reionization import load_reionization_history
    from .ndot import NdotCalculator
    
    z = test.get('z', 5.0)
    gamma = test.get('gamma', 0.5e-12)
    
    history_file = test.get('history')
    if history_file:
        history = load_reionization_history(path=history_file, model='file')
    else:
        z_re = test.get('z_re', 7.0)
        Delta_z = test.get('Delta_z', 1.5)
        history = load_reionization_history(model='tanh', z_re=z_re, Delta_z=Delta_z)
    
    try:
        calc = NdotCalculator(emulator, history)
        ndot = calc.compute_ndot_emulator(z, gamma)
        
        if not (ndot is None or (hasattr(ndot, '__iter__') and len(ndot) == 0)):
            import numpy as np
            if not np.isnan(ndot) and ndot > 0:
                return {'success': True, 'message': f"ndot={ndot:.2e} at z={z}"}
        
        return {'success': False, 'message': f"Invalid ndot result at z={z}"}
        
    except Exception as e:
        return {'success': False, 'message': f"ndot calculation failed: {str(e)}"}


if __name__ == '__main__':
    main()
