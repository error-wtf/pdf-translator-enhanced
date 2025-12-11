# pdf-translator
english scientific PDF to LaTeX and back to PDF translator


## Automated Document Translation and Layout Preservation Framework for Scientific Literature

### I. Introduction

This document details the architecture and methodology of an automated framework designed for the **robust translation of scientific documents** (specifically PDFs) while preserving their original structural and mathematical formatting. The core challenge addressed is the translation of complex, multi-modal content (text, equations, tables, figures) into a target language while generating a final document with a layout suitable for academic dissemination. The proposed solution employs an asynchronous **microservice architecture** utilizing Large Language Models (LLMs) for high-quality translation and the $\mathbf{\LaTeX}$ typesetting system for precise output reconstruction.

***

### II. System Architecture and Methodology

The framework operates via a multi-stage pipeline, managed asynchronously to ensure system responsiveness. The process is orchestrated through a RESTful API and managed by dedicated job-handling modules.

#### A. Job Management and State Tracking

The system utilizes an in-memory job registry (`jobs.py`) to manage the lifecycle of each translation request.

| Component | Function | Data Structure |
| :--- | :--- | :--- |
| **API Endpoint** (`/upload`) | Initiates the job, saves the original PDF, and starts an **asynchronous thread** (`start_job_thread`). | `JobInfo` (via `models.py`) |
| **State Enumeration** | Tracks progress through defined states. | `JobStatus` (queued, analyzing, translating, latex\_build, done, error) |
| **Status Update** | Provides granular progress and status messages to the client. | `progress` (0-100), `message` |

**Data Model:** The primary unit of content is the **Block** (`models.py`), defined by:
* `page`: The page number of origin.
* `content`: The extracted source text.
* `translated_latex`: The translated and $\LaTeX$-ready text.
* `element_type`: Categorizes the block for layout reconstruction (e.g., **text**, **figure\_caption**, **table\_content**, **image\_placeholder**).
* `bbox`: Bounding Box (x0, y0, x1, y1) for layout-aware processing.

#### B. Structural Analysis (PDF Pre-processing)

The structural analysis (`pdf_processing.py`) is designed to decompose the source PDF into a sequence of meaningful `Block` objects, enabling targeted translation and preservation of document flow.

1.  **Text and Word Extraction:** The PDF is parsed (conceptually via a library like `pdfplumber` or `PDFMiner.six`) to extract text and associated **Bounding Boxes (BBox)** for each word.
2.  **Block Aggregation:** Adjacent words are aggregated into `Block` objects based on heuristic tolerances, specifically **vertical gaps** ($> 1.5$ units) and minimal **horizontal displacement** ($\pm 10$ units) to maintain column and paragraph integrity.
3.  **Element Classification:** Each generated block is assigned an `element_type` based on its content or contextual cues (e.g., detecting text blocks that reference figures to insert an `image_placeholder` block).
4.  **Language Detection:** A lightweight language detection utility (`langdetect`) is applied to a text sample to determine the **source language** for the translation prompt.

#### C. Machine Translation

The system utilizes an external LLM (specified as **OpenAI GPT-4.1**) acting as a **professional scientific translator**.

1.  **Prompt Engineering:** A highly structured prompt is used to enforce translation constraints, emphasizing:
    * **Strict preservation of $\mathbf{\LaTeX}$ math environments** (e.g., inline `$…$`, $\backslash(…\backslash)$, $\backslash[…\backslash]$, and `equation` environments).
    * Maintenance of all document hierarchy (headings, sections, paragraphs, formatting).
    * Fluency in the target scientific language.
2.  **Execution:** The raw block content is passed to the LLM. The resulting translated text is stored in the `translated_latex` field of a new `Block` instance.

#### D. $\mathbf{\LaTeX}$ Output Generation and Compilation

The final document assembly is handled by the $\LaTeX$ compilation module (`latex_build.py`).

1.  **Source Generation:** The `render_latex` function constructs the `main.tex` source file.
    * A standard scientific document class is utilized (`\documentclass{article}`).
    * The `babel` package is configured for proper hyphenation and localization based on the `target_language`.
    * The translated $\mathbf{Block}$ content is iterated over and formatted according to its `element_type`:
        * **Text:** Followed by two newlines (`\n\n`) to create a new paragraph.
        * **Captions/Tables/Placeholders:** Wrapped in appropriate $\mathbf{\LaTeX}$ environments (e.g., `\begin{table}`, `\begin{figure}`, or inserted directly if a pre-formatted placeholder).
2.  **Robust Compilation:** The $\mathbf{\LaTeX}$ source is compiled using `pdflatex` in a non-stop mode. The compilation is executed **twice** to ensure resolution of cross-references, table of contents, and bibliography citations.
3.  **Error Handling:** The system explicitly checks the exit code of `pdflatex`. If compilation fails, the main error message (starting with `!`) is extracted from the generated `.log` file and relayed as the job failure reason.

***

### III. Conclusion

This framework establishes a reliable, multi-stage process for high-fidelity translation of scientific PDFs. By separating the document into semantically and structurally distinct blocks, leveraging the linguistic and contextual power of modern LLMs, and utilizing the robust typesetting capabilities of $\mathbf{\LaTeX}$, the system minimizes layout degradation and maximizes the academic utility of the translated output.
