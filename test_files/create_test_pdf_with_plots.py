#!/usr/bin/env python3
"""
Create a test PDF with real matplotlib plots.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
from io import BytesIO

def create_ssz_plot():
    """Create SSZ vs GR time dilation plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Data
    r_rs = np.linspace(1.01, 10, 100)
    phi = 1.618
    
    # GR time dilation
    D_GR = np.sqrt(1 - 1/r_rs)
    
    # SSZ time dilation (strong field)
    Xi = 1 - np.exp(-phi * r_rs)
    D_SSZ = 1 / (1 + Xi)
    
    ax.plot(r_rs, D_GR, 'r-', linewidth=2, label='GR (Schwarzschild)')
    ax.plot(r_rs, D_SSZ, 'b-', linewidth=2, label='SSZ')
    ax.axvline(x=1.387, color='green', linestyle='--', label='r*/rs = 1.387')
    
    ax.set_xlabel('r/rs', fontsize=12)
    ax.set_ylabel('D (Time Dilation Factor)', fontsize=12)
    ax.set_title('Time Dilation: SSZ vs General Relativity', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 1)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def create_power_law_plot():
    """Create power law fit plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Simulated data points
    np.random.seed(42)
    r_s_R = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    E_ratio = 1 + 0.32 * r_s_R**0.98 + np.random.normal(0, 0.01, len(r_s_R))
    
    # Fit line
    x_fit = np.linspace(0, 1, 100)
    y_fit = 1 + 0.32 * x_fit**0.98
    
    ax.scatter(r_s_R, E_ratio, s=80, c='blue', marker='o', label='Data points', zorder=5)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit: E/E₀ = 1 + 0.32(rs/R)^0.98')
    
    ax.set_xlabel('rs/R', fontsize=12)
    ax.set_ylabel('E/E_rest', fontsize=12)
    ax.set_title('Universal Power Law (R² = 0.997)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def create_validation_plot():
    """Create validation results bar chart."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    repos = ['ssz-qubits', 'ssz-metric', 'ssz-full', 'g79-cygnus']
    tests = [74, 12, 41, 14]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(repos, tests, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, tests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Tests', fontsize=12)
    ax.set_title('Test Results by Repository (100% Pass)', fontsize=14)
    ax.set_ylim(0, 85)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def create_test_pdf_with_plots():
    """Create PDF with real plots."""
    doc = fitz.open()
    
    # Generate plots
    print("Generating plots...")
    plot1_data = create_ssz_plot()
    plot2_data = create_power_law_plot()
    plot3_data = create_validation_plot()
    
    # === PAGE 1 ===
    page1 = doc.new_page(width=595, height=842)
    
    page1.insert_text((50, 40), "SSZ Theory - Test Document with Real Plots", fontsize=16, fontname="helv")
    page1.insert_text((50, 60), "Page 1: Time Dilation Comparison", fontsize=11, fontname="helv")
    
    # Insert plot 1
    img_rect1 = fitz.Rect(50, 80, 350, 320)
    page1.insert_image(img_rect1, stream=plot1_data)
    
    # Caption
    page1.insert_text((50, 335), "Figure 1: Time dilation factor D as a function of r/rs.", fontsize=10, fontname="helv")
    page1.insert_text((50, 350), "Blue: SSZ prediction. Red: General Relativity.", fontsize=10, fontname="helv")
    
    # Text below
    text1 = """The plot above shows the fundamental difference between SSZ and GR predictions.
At the universal intersection point r*/rs = 1.387, both theories give identical results.
For r < r*, SSZ predicts higher time dilation (slower clocks).
For r > r*, SSZ predicts lower time dilation.

Key observation: At the event horizon (r = rs), GR predicts D = 0 (infinite time dilation),
while SSZ predicts D = 0.555 (finite, measurable value).

This resolves the singularity problem without introducing new physics."""
    
    page1.insert_textbox(fitz.Rect(50, 380, 545, 600), text1, fontsize=11, fontname="helv")
    
    # === PAGE 2 ===
    page2 = doc.new_page(width=595, height=842)
    
    page2.insert_text((50, 40), "Page 2: Universal Power Law", fontsize=11, fontname="helv")
    
    # Insert plot 2
    img_rect2 = fitz.Rect(50, 70, 350, 310)
    page2.insert_image(img_rect2, stream=plot2_data)
    
    # Caption
    page2.insert_text((50, 325), "Figure 2: Energy ratio E/E_rest vs compactness rs/R.", fontsize=10, fontname="helv")
    page2.insert_text((50, 340), "Fit: E/E_rest = 1 + 0.32(rs/R)^0.98 with R² = 0.997", fontsize=10, fontname="helv")
    
    text2 = """The remarkable power law relationship shown in Figure 2 was discovered through 
analysis of 129 astronomical objects including neutron stars and black holes.

The coefficient of determination R² = 0.997 indicates an excellent fit, suggesting 
this is a fundamental relationship in the SSZ framework.

Applications include:
- GPS timing corrections
- Pulsar timing analysis  
- Black hole mass estimation
- Gravitational wave parameter extraction"""
    
    page2.insert_textbox(fitz.Rect(50, 370, 545, 580), text2, fontsize=11, fontname="helv")
    
    # === PAGE 3 ===
    page3 = doc.new_page(width=595, height=842)
    
    page3.insert_text((50, 40), "Page 3: Validation Results", fontsize=11, fontname="helv")
    
    # Insert plot 3
    img_rect3 = fitz.Rect(50, 70, 350, 310)
    page3.insert_image(img_rect3, stream=plot3_data)
    
    # Caption
    page3.insert_text((50, 325), "Figure 3: Test results across SSZ repositories.", fontsize=10, fontname="helv")
    page3.insert_text((50, 340), "Total: 141 tests, 100% pass rate.", fontsize=10, fontname="helv")
    
    text3 = """The SSZ framework has been extensively validated through automated testing.

Repository breakdown:
- ssz-qubits: 74 tests (quantum computing applications)
- ssz-metric-pure: 12 tests (mathematical foundations)
- ssz-full-metric: 41 tests (astrophysical observables)
- g79-cygnus: 14 tests (Cygnus X-1 validation)

Additional validation:
- 47 ESO spectroscopy measurements: 97.9% accuracy
- Mercury perihelion precession: 99.67% match
- GPS timing: sub-nanosecond agreement

All code available at: https://github.com/error-wtf/"""
    
    page3.insert_textbox(fitz.Rect(50, 370, 545, 620), text3, fontsize=11, fontname="helv")
    
    # Save
    output_path = Path(__file__).parent.parent / "test_with_plots.pdf"
    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    create_test_pdf_with_plots()
