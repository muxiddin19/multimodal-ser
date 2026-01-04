"""
Analyze preference survey data for ACL 2026.

Generates:
1. Per-user learning curves (accuracy vs. number of comparisons)
2. Stratified results by demographics
3. Statistical tests comparing baselines

Usage:
    python analyze_preferences.py --data_dir data/
"""

import json
import os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, bootstrap
import matplotlib.pyplot as plt


def load_data(data_dir):
    """Load all user profiles and comparisons."""
    users = {}
    comparisons = defaultdict(list)

    for f in os.listdir(data_dir):
        if f.endswith('_profile.json'):
            user_id = f.replace('_profile.json', '')
            with open(os.path.join(data_dir, f)) as fp:
                users[user_id] = json.load(fp)

        elif f.endswith('_comparisons.jsonl'):
            user_id = f.replace('_comparisons.jsonl', '')
            with open(os.path.join(data_dir, f)) as fp:
                for line in fp:
                    comparisons[user_id].append(json.loads(line))

    return users, comparisons


def compute_learning_curves(users, comparisons, train_sizes=[2, 4, 8, 12, 16, 20, 24]):
    """
    Compute accuracy as a function of training comparisons.

    For each user:
    1. Train on first N comparisons
    2. Test on remaining comparisons
    3. Report accuracy

    This answers the reviewer question about per-user learning curves.
    """
    results = {n: [] for n in train_sizes}

    for user_id, comps in comparisons.items():
        if len(comps) < max(train_sizes) + 4:  # Need some test data
            continue

        # Build user-specific preference model
        for n_train in train_sizes:
            if n_train >= len(comps) - 2:
                continue

            train_comps = comps[:n_train]
            test_comps = comps[n_train:]

            # Simple logistic model: predict winner from style features
            X_train, y_train = [], []
            for c in train_comps:
                # Features: one-hot for styles
                feat = [0] * 12  # 6 styles for each option
                feat[c['style_a']] = 1
                feat[6 + c['style_b']] = 1
                X_train.append(feat)
                y_train.append(1 if c['winner'] == c['style_a'] else 0)

            if len(set(y_train)) < 2:
                continue

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Test accuracy
            correct = 0
            for c in test_comps:
                feat = [0] * 12
                feat[c['style_a']] = 1
                feat[6 + c['style_b']] = 1
                pred = model.predict([feat])[0]
                actual = 1 if c['winner'] == c['style_a'] else 0
                if pred == actual:
                    correct += 1

            acc = correct / len(test_comps) if test_comps else 0
            results[n_train].append(acc)

    return results


def stratified_analysis(users, comparisons):
    """
    Analyze results stratified by user attributes.

    Answers reviewer question about demographic stratification.
    """
    stratified = {
        'by_age': defaultdict(list),
        'by_culture': defaultdict(list),
        'by_accessibility': {'with_needs': [], 'without_needs': []}
    }

    for user_id, profile in users.items():
        if user_id not in comparisons:
            continue

        comps = comparisons[user_id]
        if len(comps) < 10:
            continue

        # Train on first half, test on second half
        n_train = len(comps) // 2
        train_comps = comps[:n_train]
        test_comps = comps[n_train:]

        X_train, y_train = [], []
        for c in train_comps:
            feat = [0] * 12
            feat[c['style_a']] = 1
            feat[6 + c['style_b']] = 1
            X_train.append(feat)
            y_train.append(1 if c['winner'] == c['style_a'] else 0)

        if len(set(y_train)) < 2:
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        correct = 0
        for c in test_comps:
            feat = [0] * 12
            feat[c['style_a']] = 1
            feat[6 + c['style_b']] = 1
            pred = model.predict([feat])[0]
            actual = 1 if c['winner'] == c['style_a'] else 0
            if pred == actual:
                correct += 1

        acc = correct / len(test_comps)

        # Stratify
        stratified['by_age'][profile.get('age_group', 'unknown')].append(acc)
        stratified['by_culture'][profile.get('culture', 'unknown')].append(acc)

        has_needs = (profile.get('low_vision') or
                     profile.get('color_blind') or
                     profile.get('dyslexia'))
        if has_needs:
            stratified['by_accessibility']['with_needs'].append(acc)
        else:
            stratified['by_accessibility']['without_needs'].append(acc)

    return stratified


def plot_learning_curves(results, output_path='learning_curves.png'):
    """Plot per-user learning curves."""
    plt.figure(figsize=(8, 5))

    train_sizes = sorted(results.keys())
    means = [np.mean(results[n]) * 100 for n in train_sizes if results[n]]
    stds = [np.std(results[n]) * 100 for n in train_sizes if results[n]]
    train_sizes = [n for n in train_sizes if results[n]]

    plt.errorbar(train_sizes, means, yerr=stds, marker='o', capsize=5,
                 linewidth=2, markersize=8)

    plt.axhline(y=50, color='gray', linestyle='--', label='Random')
    plt.xlabel('Number of Training Comparisons', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Preference Learning Curves', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(40, 80)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Learning curves saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    args = parser.parse_args()

    print("Loading data...")
    users, comparisons = load_data(args.data_dir)

    print(f"Loaded {len(users)} users with {sum(len(c) for c in comparisons.values())} comparisons")

    if len(users) < 5:
        print("\nNot enough data yet. Run the survey to collect more responses.")
        print("Start the survey with: python preference_survey.py")
        return

    print("\n" + "="*60)
    print("Learning Curve Analysis")
    print("="*60)

    results = compute_learning_curves(users, comparisons)

    for n, accs in sorted(results.items()):
        if accs:
            print(f"  {n} comparisons: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% (n={len(accs)})")

    plot_learning_curves(results)

    print("\n" + "="*60)
    print("Stratified Analysis")
    print("="*60)

    stratified = stratified_analysis(users, comparisons)

    print("\nBy Age Group:")
    for age, accs in stratified['by_age'].items():
        if accs:
            print(f"  {age}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% (n={len(accs)})")

    print("\nBy Culture:")
    for culture, accs in stratified['by_culture'].items():
        if accs:
            print(f"  {culture}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% (n={len(accs)})")

    print("\nBy Accessibility Needs:")
    for group, accs in stratified['by_accessibility'].items():
        if accs:
            print(f"  {group}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% (n={len(accs)})")

    # Statistical test
    with_needs = stratified['by_accessibility']['with_needs']
    without_needs = stratified['by_accessibility']['without_needs']
    if len(with_needs) > 2 and len(without_needs) > 2:
        t_stat, p_val = ttest_ind(with_needs, without_needs)
        print(f"\n  t-test (with vs without needs): t={t_stat:.2f}, p={p_val:.3f}")

    # Save summary
    summary = {
        'n_users': len(users),
        'n_comparisons': sum(len(c) for c in comparisons.values()),
        'learning_curves': {
            str(n): {
                'mean': float(np.mean(accs)) if accs else None,
                'std': float(np.std(accs)) if accs else None,
                'n': len(accs)
            }
            for n, accs in results.items()
        },
        'stratified': {
            'by_age': {
                age: {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'n': len(accs)}
                for age, accs in stratified['by_age'].items() if accs
            },
            'by_culture': {
                cult: {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'n': len(accs)}
                for cult, accs in stratified['by_culture'].items() if accs
            }
        }
    }

    output_path = os.path.join(args.data_dir, 'analysis_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_path}")


if __name__ == '__main__':
    main()
