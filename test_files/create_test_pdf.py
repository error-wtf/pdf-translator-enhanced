#!/usr/bin/env python3
"""
Create a test PDF with 3 pages containing:
- Page 1: Scientific text with LaTeX formulas
- Page 2: Text with bullet points and tables
- Page 3: Text with images (placeholder rectangles)
"""

import fitz  # PyMuPDF

def create_test_pdf():
    doc = fitz.open()
    
    # === PAGE 1: Scientific text with formulas ===
    page1 = doc.new_page(width=595, height=842)  # A4
    
    # Title
    page1.insert_text((50, 50), "SSZ Theory - Scientific Document", fontsize=18, fontname="helv")
    page1.insert_text((50, 70), "Page 1: Formulas and Scientific Text", fontsize=12, fontname="helv")
    
    # Scientific text
    text1 = """Abstract

The Segmented Spacetime Zeta (SSZ) theory proposes a modification to general 
relativity that resolves the singularity problem at black hole horizons.

The fundamental equation is E = mc², which relates energy and mass.

Key Formulas:

The time dilation factor is given by D_SSZ = 1/(1 + Xi), where Xi is the 
segment density function.

For weak fields (r/r_s > 100):
Xi(r) = r_s / (2r)

For strong fields (r/r_s < 100):
Xi(r) = 1 - exp(-phi * r / r_s)

where phi = 1.618... is the golden ratio.

The universal intersection point occurs at r*/r_s = 1.387 ± 0.002.

Results show 99.1% accuracy compared to ESO spectroscopy data across 
47 astronomical objects including neutron stars and black holes.
"""
    
    page1.insert_textbox(
        fitz.Rect(50, 100, 545, 750),
        text1,
        fontsize=11,
        fontname="helv"
    )
    
    # === PAGE 2: Text with bullet points ===
    page2 = doc.new_page(width=595, height=842)
    
    page2.insert_text((50, 50), "SSZ Theory - Applications", fontsize=18, fontname="helv")
    page2.insert_text((50, 70), "Page 2: Lists and Structure", fontsize=12, fontname="helv")
    
    text2 = """Introduction

The SSZ theory has several important applications:

- Quantum Computing: Gate timing corrections for superconducting qubits
- Astrophysics: Black hole shadow predictions for EHT observations
- GPS Systems: Improved timing accuracy for satellite navigation
- Gravitational Waves: Modified ringdown frequencies for LIGO

Validation Results:

Repository           Tests    Status
-----------------------------------------
ssz-qubits           74       PASSED
ssz-metric-pure      12       PASSED  
ssz-full-metric      41       PASSED
g79-cygnus           14       PASSED
-----------------------------------------
Total                141      100%

The validation uses professional ESO spectroscopy measurements with 
97.9% accuracy on 47 real astronomical objects.

Data Availability:

All code, data, and documentation are publicly available:

- SSZ-Qubits: https://github.com/error-wtf/ssz-qubits
- SSZ-Metric-Pure: https://github.com/error-wtf/ssz-metric-pure
- SSZ-Full-Metric: https://github.com/error-wtf/ssz-metric-final
"""
    
    page2.insert_textbox(
        fitz.Rect(50, 100, 545, 750),
        text2,
        fontsize=11,
        fontname="helv"
    )
    
    # === PAGE 3: Text with image placeholders ===
    page3 = doc.new_page(width=595, height=842)
    
    page3.insert_text((50, 50), "SSZ Theory - Visualizations", fontsize=18, fontname="helv")
    page3.insert_text((50, 70), "Page 3: Figures and Images", fontsize=12, fontname="helv")
    
    # Draw placeholder rectangles for "images"
    # Figure 1
    rect1 = fitz.Rect(50, 100, 280, 250)
    page3.draw_rect(rect1, color=(0.7, 0.7, 0.7), fill=(0.9, 0.9, 0.9))
    page3.insert_text((100, 180), "Figure 1", fontsize=10, fontname="helv")
    page3.insert_text((50, 265), "Fig. 1: Time dilation comparison SSZ vs GR", fontsize=9, fontname="helv")
    
    # Figure 2
    rect2 = fitz.Rect(310, 100, 545, 250)
    page3.draw_rect(rect2, color=(0.7, 0.7, 0.7), fill=(0.9, 0.9, 0.9))
    page3.insert_text((370, 180), "Figure 2", fontsize=10, fontname="helv")
    page3.insert_text((310, 265), "Fig. 2: Universal power law R² = 0.997", fontsize=9, fontname="helv")
    
    text3 = """Discussion

Figure 1 shows the time dilation factor D as a function of radial distance r/r_s. 
The SSZ prediction (blue) differs from GR (red) in the strong field regime but 
converges at the universal intersection point r*/r_s = 1.387.

Figure 2 demonstrates the remarkable power law relationship E/E_rest = 1 + 0.32(r_s/R)^0.98 
with coefficient of determination R² = 0.997, indicating an excellent fit to the data.

Conclusions

The SSZ theory provides:

1. Resolution of the singularity problem at black hole horizons
2. Finite time dilation at r = r_s (D = 0.555 vs D = 0 in GR)
3. Universal intersection point independent of black hole mass
4. Excellent agreement with observational data (99.1% accuracy)

Future work will focus on:
- Next-generation Event Horizon Telescope observations (2027-2030)
- NICER neutron star measurements (2025-2027)
- Quantum computing applications with IBM and Google hardware
"""
    
    page3.insert_textbox(
        fitz.Rect(50, 290, 545, 750),
        text3,
        fontsize=11,
        fontname="helv"
    )
    
    # Save
    output_path = "test_3pages.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_test_pdf()
