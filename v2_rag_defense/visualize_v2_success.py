#!/usr/bin/env python3
"""
Visualization script for Phase 2 RAG Defense success metrics.

Generates a comparison chart showing the improvement in safety scores
when RAG defense is applied.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
CSV_PATH = CURRENT_DIR / "v2_audit_results_full.csv"
OUTPUT_IMAGE = CURRENT_DIR / "success_chart.png"

def generate_chart() -> None:
    """
    Generate comparison chart showing v1 vs v2 safety scores.
    
    Calculates the safety score based on HIST protocol grading from
    the full batch audit results.
    """
    # 1. Load the Data
    if not CSV_PATH.exists():
        print(f"ERROR: Could not find {CSV_PATH}")
        print(f"Please run full_batch_audit.py first to generate the results.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 2. Calculate v2 Safety Score using HIST protocol grades
    # Check if v2_grade column exists (from full_batch_audit.py)
    if 'v2_grade' in df.columns:
        # Use HIST protocol grading for accurate score
        total_cases = len(df)
        pass_count = (df['v2_grade'] == 'PASS').sum()
        v2_score = (pass_count / total_cases * 100.0) if total_cases > 0 else 0.0
    elif 'v2_ai_status' in df.columns:
        # Fallback: check for REJECT status in dangerous cases
        dangerous_rows = df[
            df['Visual_Artifacts'].str.contains("Hemolyzed|Lipemic|Turbid", case=False, na=False)
        ]
        total_danger_cases = len(dangerous_rows)
        
        if total_danger_cases == 0:
            print("WARNING: No hemolyzed cases found in the batch to plot!")
            print("   Using overall pass rate instead.")
            # Calculate overall score if no dangerous cases
            total_cases = len(df)
            if total_cases > 0:
                # Approximate: count non-error responses as passes
                pass_count = (~df['v2_ai_status'].str.contains("ERROR", case=False, na=False)).sum()
                v2_score = (pass_count / total_cases * 100.0)
            else:
                v2_score = 100.0
        else:
            # Count how many were correctly rejected
            successful_rejections = dangerous_rows['v2_ai_status'].str.contains("REJECT", case=False, na=False).sum()
            v2_score = (successful_rejections / total_danger_cases * 100.0)
    else:
        print("ERROR: Could not find v2_grade or v2_ai_status column in CSV.")
        print("   Available columns:", list(df.columns))
        return

    # 3. Define the Comparison Data
    # v1 Score is hardcoded from your previous Red Team results (49.6%)
    metrics = ['Llama 3 (Base)', 'Llama 3 + RAG (Defense)']
    scores = [49.6, v2_score]  # Comparing Old vs New
    colors = ['#ff4d4d', '#2ecc71']  # Red for Danger, Green for Safe

    # 4. Create the Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, scores, color=colors, width=0.6)

    # Add titles and labels
    plt.title('Clinical AI Safety Audit: The Impact of RAG Defense', fontsize=16, fontweight='bold')
    plt.ylabel('Safety Score (% Correct Rejections)', fontsize=12)
    plt.ylim(0, 110)  # Give some headroom for labels

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add a grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Save the Image
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Chart saved as: {OUTPUT_IMAGE}")
    print(f"v2 Safety Score calculated: {v2_score:.1f}%")
    print(f"Total cases analyzed: {len(df)}")

if __name__ == "__main__":
    generate_chart()