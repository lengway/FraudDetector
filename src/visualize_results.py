"""
Visualization Script for Fraud Detection Results
=================================================

Generates:
1. Confusion Matrix
2. ROC Curve with AUC
3. Feature Importance Chart
4. Score Distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from catboost import CatBoostClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def load_results():
    """Загрузка результатов детекции."""
    df = pd.read_csv('docs/two_stage_detection_results.csv')
    print(f"✅ Loaded {len(df)} transactions")
    print(f"   Fraud rate: {df['target'].mean()*100:.2f}%")
    return df


def plot_confusion_matrix(df, save_path='docs/confusion_matrix.png'):
    """Построение Confusion Matrix."""
    
    # Создаем confusion matrix
    y_true = df['target']
    y_pred = df['fraud_prediction']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Расчет метрик
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Нормализованная матрица для отображения
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Цветовая карта
    colors = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    # Heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Fraud (0)', 'Fraud (1)'],
                yticklabels=['Not Fraud (0)', 'Fraud (1)'])
    
    # Добавляем текст с числами и процентами
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            label = {
                (0, 0): 'TN',
                (0, 1): 'FP',
                (1, 0): 'FN',
                (1, 1): 'TP'
            }[(i, j)]
            color = 'white' if count > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{label}\n{count:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color=color)
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix - Two-Stage Fraud Detection', fontsize=16, fontweight='bold')
    
    # Добавляем метрики на график
    metrics_text = (
        f'Accuracy: {accuracy:.1%}\n'
        f'Precision: {precision:.1%}\n'
        f'Recall: {recall:.1%}\n'
        f'F1 Score: {f1:.1%}'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion Matrix saved to '{save_path}'")
    print(f"   TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"   Recall: {recall:.1%}, Precision: {precision:.1%}, F1: {f1:.1%}")
    
    return cm


def plot_roc_curve(save_path='docs/roc_curve.png'):
    """Построение ROC-кривой с использованием ML модели."""
    
    # Загружаем модель и данные
    model = CatBoostClassifier()
    model.load_model('models/catboost_fraud_model.cbm')
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Загружаем результаты с вероятностями
    df = pd.read_csv('docs/two_stage_detection_results.csv')
    
    # Только ML-проверенные транзакции (scorecard_total > 1)
    df_ml = df[df['scorecard_total'] > 1].copy()
    
    y_true = df_ml['target']
    y_prob = df_ml['fraud_probability']
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Находим оптимальный порог (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC Curve
    ax.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    
    # Диагональ (random classifier)
    ax.plot([0, 1], [0, 1], color='#888888', lw=2, linestyle='--', label='Random Classifier')
    
    # Оптимальная точка
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=200, c='red', marker='*', 
               zorder=5, label=f'Optimal Threshold = {optimal_threshold:.2f}')
    
    # Текущий порог (0.20)
    current_threshold = 0.20
    current_idx = np.argmin(np.abs(thresholds - current_threshold))
    ax.scatter(fpr[current_idx], tpr[current_idx], s=150, c='green', marker='o',
               zorder=5, label=f'Current Threshold = {current_threshold}')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=14)
    ax.set_title('ROC Curve - CatBoost Fraud Detection Model', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    # Аннотации
    ax.annotate(f'AUC = {roc_auc:.3f}', xy=(0.6, 0.2), fontsize=16, 
                fontweight='bold', color='#2E86AB')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC Curve saved to '{save_path}'")
    print(f"   AUC-ROC: {roc_auc:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.3f}")
    
    return roc_auc


def plot_feature_importance(top_n=15, save_path='docs/feature_importance.png'):
    """Построение графика важности фичей."""
    
    # Загружаем модель
    model = CatBoostClassifier()
    model.load_model('models/catboost_fraud_model.cbm')
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Важность фичей
    importance = model.get_feature_importance()
    
    # Создаем DataFrame и сортируем
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(top_n)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_importance)))
    
    bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    # Добавляем значения на бары
    for bar, val in zip(bars, feature_importance['importance']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Feature Importance (%)', fontsize=14)
    ax.set_title(f'Top {top_n} Most Important Features - CatBoost Model', 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(feature_importance['importance']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature Importance chart saved to '{save_path}'")
    print(f"   Top 3 features:")
    for i, row in feature_importance.tail(3).iloc[::-1].iterrows():
        print(f"      {row['feature']}: {row['importance']:.2f}%")
    
    return feature_importance


def plot_score_distribution(save_path='docs/score_distribution.png'):
    """Распределение скорекарда и вероятностей фрода."""
    
    df = pd.read_csv('docs/two_stage_detection_results.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Распределение scorecard баллов
    ax1 = axes[0]
    
    fraud_scores = df[df['target'] == 1]['scorecard_total']
    legit_scores = df[df['target'] == 0]['scorecard_total']
    
    bins = range(0, int(max(df['scorecard_total'])) + 3)
    
    ax1.hist(legit_scores, bins=bins, alpha=0.7, label='Legitimate', color='#2ECC71', edgecolor='black')
    ax1.hist(fraud_scores, bins=bins, alpha=0.7, label='Fraud', color='#E74C3C', edgecolor='black')
    
    ax1.axvline(x=1.5, color='blue', linestyle='--', linewidth=2, label='ML Threshold (score>1)')
    
    ax1.set_xlabel('Scorecard Score', fontsize=12)
    ax1.set_ylabel('Number of Transactions', fontsize=12)
    ax1.set_title('Scorecard Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Распределение вероятностей фрода (только ML-проверенные)
    ax2 = axes[1]
    
    df_ml = df[df['scorecard_total'] > 1]
    
    fraud_prob = df_ml[df_ml['target'] == 1]['fraud_probability']
    legit_prob = df_ml[df_ml['target'] == 0]['fraud_probability']
    
    bins = np.linspace(0, 1, 30)
    
    ax2.hist(legit_prob, bins=bins, alpha=0.7, label='Legitimate', color='#2ECC71', edgecolor='black')
    ax2.hist(fraud_prob, bins=bins, alpha=0.7, label='Fraud', color='#E74C3C', edgecolor='black')
    
    ax2.axvline(x=0.20, color='blue', linestyle='--', linewidth=2, label='Classification Threshold (0.20)')
    
    ax2.set_xlabel('Fraud Probability', fontsize=12)
    ax2.set_ylabel('Number of Transactions', fontsize=12)
    ax2.set_title('Fraud Probability Distribution (ML-Checked)', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.suptitle('Two-Stage Fraud Detection: Score Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Score Distribution saved to '{save_path}'")


def create_summary_dashboard(save_path='docs/fraud_detection_dashboard.png'):
    """Создание общего дашборда со всеми визуализациями."""
    
    df = pd.read_csv('docs/two_stage_detection_results.csv')
    
    # Загружаем модель для ROC
    model = CatBoostClassifier()
    model.load_model('models/catboost_fraud_model.cbm')
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Метрики
    y_true = df['target']
    y_pred = df['fraud_prediction']
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Layout: 2x2 grid
    
    # 1. Confusion Matrix (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            label = {(0,0): 'TN', (0,1): 'FP', (1,0): 'FN', (1,1): 'TP'}[(i, j)]
            color = 'white' if count > cm.max() / 2 else 'black'
            ax1.text(j + 0.5, i + 0.5, f'{label}\n{count:,}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. ROC Curve (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    df_ml = df[df['scorecard_total'] > 1]
    y_true_ml = df_ml['target']
    y_prob_ml = df_ml['fraud_probability']
    
    fpr, tpr, thresholds = roc_curve(y_true_ml, y_prob_ml)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    importance = model.get_feature_importance()
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True).tail(10)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_imp)))
    ax3.barh(feature_imp['feature'], feature_imp['importance'], color=colors)
    ax3.set_xlabel('Importance (%)')
    ax3.set_title('Top 10 Features', fontsize=14, fontweight='bold')
    
    # 4. Key Metrics (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    metrics_text = (
        f"═══════════════════════════════════\n"
        f"    FRAUD DETECTION METRICS\n"
        f"═══════════════════════════════════\n\n"
        f"  📊 Dataset Statistics:\n"
        f"     • Total Transactions: {len(df):,}\n"
        f"     • Fraud Cases: {int(df['target'].sum()):,} ({df['target'].mean()*100:.2f}%)\n\n"
        f"  🎯 Model Performance:\n"
        f"     • Recall: {recall:.1%} (caught {tp} of {tp+fn} frauds)\n"
        f"     • Precision: {precision:.1%}\n"
        f"     • F1 Score: {f1:.1%}\n"
        f"     • AUC-ROC: {roc_auc:.3f}\n\n"
        f"  ⚠️ Error Analysis:\n"
        f"     • False Positives: {fp} (false alarms)\n"
        f"     • False Negatives: {fn} (missed frauds)\n\n"
        f"  🏗️ Architecture:\n"
        f"     • Stage 1: Scorecard (25 rules)\n"
        f"     • Stage 2: CatBoost ML ({len(feature_names)} features)\n"
        f"═══════════════════════════════════"
    )
    
    ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.suptitle('Two-Stage Fraud Detection System - Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Dashboard saved to '{save_path}'")


def main():
    """Генерация всех визуализаций."""
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Загрузка данных
    df = load_results()
    
    # 1. Confusion Matrix
    print("\n1️⃣ Generating Confusion Matrix...")
    plot_confusion_matrix(df)
    
    # 2. ROC Curve
    print("\n2️⃣ Generating ROC Curve...")
    plot_roc_curve()
    
    # 3. Feature Importance
    print("\n3️⃣ Generating Feature Importance...")
    plot_feature_importance()
    
    # 4. Score Distribution
    print("\n4️⃣ Generating Score Distribution...")
    plot_score_distribution()
    
    # 5. Summary Dashboard
    print("\n5️⃣ Generating Summary Dashboard...")
    create_summary_dashboard()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print("\nFiles saved in 'docs/' folder:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • feature_importance.png")
    print("  • score_distribution.png")
    print("  • fraud_detection_dashboard.png (MAIN)")
    print("="*60)


if __name__ == '__main__':
    main()
