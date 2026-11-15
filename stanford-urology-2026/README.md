<div align="center">

<img src="assets/banner.svg" alt="Functional Imaging of Pelvic Floor Muscles" width="100%">

<br>

![Status](https://img.shields.io/badge/status-preliminary_findings-f59e0b?style=for-the-badge)
![Imaging](https://img.shields.io/badge/imaging-perineal_ultrasound-0ea5e9?style=for-the-badge)
![Subjects](https://img.shields.io/badge/subjects-23-10b981?style=for-the-badge)
![Probe](https://img.shields.io/badge/probe-5.5_MHz-8b5cf6?style=for-the-badge)
![Mentor](https://img.shields.io/badge/mentor-Stanford_Prof._Emeritus-8c1515?style=for-the-badge)

**How does the pelvic floor move when we contract, push, and cough?**
A mentored biomedical-imaging project turning ultrasound recordings into displacement, velocity and acceleration.

[🎬 Watch the clips](#-see-it-in-action) ·
[🔬 Method](#-how-the-analysis-works) ·
[📊 Findings](#-preliminary-findings) ·
[🎤 Slides](#-the-research-deck) ·
[🚀 Roadmap](#-roadmap)

</div>

---

## 👋 At a glance

| | |
|---|---|
| 👤 **Author** | Aaryan Samanta |
| 🎓 **Mentor** | [Dr. Christos E. Constantinou](https://med.stanford.edu/profiles/christos-constantinou), Professor Emeritus, Stanford Surgery (Urology) & Human Biology |
| 🧬 **Focus** | Kinematic ultrasound imaging of the female pelvic floor |
| 🗓️ **Timeline** | Phase 1: Dec 2025 – Mar 2026 · Phase 2: Jun 2026 |
| 📌 **Status** | Preliminary findings · last updated 2026-09-19 |

> [!IMPORTANT]
> **Scope.** This repository documents the *analysis* of anonymized clinical ultrasound data **provided by the mentor**, plus a research presentation of preliminary findings. It is **not** a publication or a manuscript, and results are preliminary. Please read [Data & privacy](docs/data-and-privacy.md) before sharing.

---

## 🌈 The journey

```mermaid
flowchart LR
    A([🎓 Mentor confirmed]):::start --> B

    subgraph P1 ["🧭 Phase 1 · Dec 2025 – Mar 2026"]
        B["Topic & data defined<br/>4 stimulation states"] --> C["Background reading<br/>pelvic floor imaging"]
    end

    C --> D

    subgraph P2 ["🔬 Phase 2 · Jun 2026"]
        D["Imaging analysis<br/>displacement · velocity · acceleration"] --> E["Figures &<br/>visual comparisons"] --> F["🎤 Research deck<br/>dated 6/22"]
    end

    F --> G(["🚧 Next: analysis log,<br/>clean figures, references"]):::next

    classDef start fill:#22d3ee,stroke:#0e7490,color:#083344
    classDef next fill:#fbbf24,stroke:#b45309,color:#451a03
    style P1 fill:#eef2ff,stroke:#6366f1,color:#312e81
    style P2 fill:#fdf2f8,stroke:#ec4899,color:#831843
```

---

## 🧭 Repository map

```text
stanford-urology-2026/
├── 📄 README.md                         ← you are here
├── 🎨 assets/                           banner + GIF previews used in READMEs
├── 📚 docs/
│   ├── project-summary.md               the whole project on one page
│   ├── references.md                    citations + reading list (DOIs)
│   └── data-and-privacy.md              data ownership, sharing & what is excluded
├── 🧭 01-phase-1-topic-and-data/        Dec 2025 – Mar 2026
│   ├── README.md                        topic, data description, goals
│   ├── reading/                         BJU 2002 (Constantinou et al.)
│   └── media/                           anatomy animation + annotated ultrasound clip
└── 🔬 02-phase-2-imaging-analysis/      Jun 2026
    ├── README.md                        sessions, deliverables, status
    ├── analysis-log.md                  image-analysis log (started)
    ├── presentation/                    research deck (PDF) + slide previews
    ├── reading/                         Miller 1998 ("The Knack")
    └── media/
        ├── ultrasound/                  3 annotated scans + frames/ (stills for slides 5–7)
        ├── pressure-maps/               5 companion pressure visualizations
        └── anatomy-animation/           3D anatomy animation (MP4)
```

| Go to… | What you'll find |
|---|---|
| [🧭 Phase 1](01-phase-1-topic-and-data/) | Why this topic, what the data are, the four stimulation states |
| [🔬 Phase 2](02-phase-2-imaging-analysis/) | The analysis workflow, deck, clips and deliverable status |
| [📚 Docs](docs/) | Summary, references and data-handling notes |

---

## 🎬 See it in action

Three test states, three scans. Colored traces follow the movement of the imaged structures.

<table>
  <tr>
    <td align="center"><b>💪 Voluntary<br>"The Knack"</b></td>
    <td align="center"><b>⬇️ Passive<br>Valsalva</b></td>
    <td align="center"><b>💨 Transient<br>Cough</b></td>
  </tr>
  <tr>
    <td><img src="assets/gifs/ultrasound-knack.gif" width="300" alt="Ultrasound during the pelvic floor maneuver"></td>
    <td><img src="assets/gifs/ultrasound-valsalva.gif" width="300" alt="Ultrasound during Valsalva"></td>
    <td><img src="assets/gifs/ultrasound-cough.gif" width="300" alt="Ultrasound during cough"></td>
  </tr>
  <tr>
    <td align="center"><sub>Contraction lifts and compresses</sub></td>
    <td align="center"><sub>Downward push tests support</sub></td>
    <td align="center"><sub>Rapid reflexive response</sub></td>
  </tr>
</table>

<sub>Full-quality MP4s: [`media/ultrasound/`](02-phase-2-imaging-analysis/media/ultrasound/)</sub>

---

## 🔬 How the analysis works

```mermaid
flowchart LR
    A["🩻 Perineal ultrasound<br/>5.5 MHz probe · 23 subjects"]:::a --> B["💾 Video recorded<br/>to a PC"]:::b
    B --> C["🧑‍⚕️ Offline analysis<br/>by a trained physician"]:::c
    C --> D["📏 Displacement<br/>urethra · anorectal junction"]:::d
    D -- "d/dt" --> E["🏃 Velocity"]:::e
    E -- "d/dt" --> F["⚡ Acceleration"]:::f

    classDef a fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    classDef b fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef c fill:#fce7f3,stroke:#db2777,color:#831843
    classDef d fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef e fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef f fill:#ffedd5,stroke:#ea580c,color:#7c2d12
```

| | |
|---|---|
| **Cohort** | 23 asymptomatic subjects, age 31.1 ± 13.6 years |
| **Method** | Perineal ultrasound, 5.5 MHz elliptical probe |
| **Segments** | Urethral displacement · anorectal junction (ARJ) motility |
| **Velocity** | Differentiating displacement over time |
| **Acceleration** | Re-differentiating velocity over time |

### 🧩 The three test states

| State | 🩺 Bladder / urethra | 🍑 Anorectal junction |
|---|---|---|
| 💪 **Voluntary**<br>(PFM pre-contraction, "The Knack") | Raises urethral closure pressure, elevates the bladder neck, prevents leakage under stress | Cranial lift and stabilization of pelvic floor structures |
| ⬇️ **Passive**<br>(Valsalva) | Bladder neck and urethra descend; higher risk of leakage / funneling | Caudal (downward) displacement and organ descent |
| 💨 **Transient**<br>(Cough) | Sudden rise in abdominal pressure; leakage if unprotected | Brief reflexive response, enhanced by pre-contraction |

> [!TIP]
> **Why "The Knack" matters.** In a randomized study of 27 women with mild-to-moderate stress urinary incontinence, contracting the pelvic floor just before and during a cough cut leakage by an average of **98.2 %** (medium cough) and **73.3 %** (deep cough) within one week ([Miller et al., 1998](docs/references.md)).

---

## 📊 Preliminary findings

<table>
<tr>
<td width="50%" valign="top">

### 💪 Contraction supports the urethra
Voluntary pelvic floor muscle contraction results in **end compression of the urethra**, which gives the urethra **greater support**.

</td>
<td width="50%" valign="top">

### 💨 No significant downward shift on cough
There is **no significant caudal (downward) displacement** during a cough.

</td>
</tr>
</table>

> [!NOTE]
> **Take-home message.** Perineal ultrasound is a non-invasive way to image dynamic pelvic floor function, and image analysis turns it into numerical parameters for passive and dynamic activation.

### 🧪 Companion pressure visualizations

Vaginal pressure (N/cm²) at superficial, middle and deep positions, comparing healthy subjects and stress-incontinence (SUI) patients.

<table>
  <tr>
    <td align="center"><b>Cough · healthy</b></td>
    <td align="center"><b>Cough · SUI patients</b></td>
  </tr>
  <tr>
    <td><img src="assets/gifs/pressure-healthy-cough.gif" width="400" alt="Pressure profiles during cough, healthy subjects"></td>
    <td><img src="assets/gifs/pressure-SUI-cough.gif" width="400" alt="Pressure profiles during cough, SUI patients"></td>
  </tr>
</table>

<p align="center">
  <img src="assets/gifs/pressure-3d-normal-vs-SUI.gif" width="640" alt="3D pressure visualization, normal subjects versus SUI patients"><br>
  <sub>3D pressure visualization · normal subjects vs. SUI patients</sub>
</p>

<sub>All five visualizations (MP4): [`media/pressure-maps/`](02-phase-2-imaging-analysis/media/pressure-maps/)</sub>

---

## 🎤 The research deck

*Functional Imaging of Pelvic Floor Muscles: Ultrasound Scanning of Asymptomatic Subjects* · draft dated 6/22 · 12 slides
**[📥 Open the full PDF](02-phase-2-imaging-analysis/presentation/Functional-Imaging-of-Pelvic-Floor-Muscles_slides_2026-06-22.pdf)**

<table>
  <tr>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-01-title.jpg" alt="Title slide"></a></td>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-02-methodology.jpg" alt="Methodology slide"></a></td>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-03-data-analysis.jpg" alt="Data collection and analysis slide"></a></td>
  </tr>
  <tr>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-04-activation-effects.jpg" alt="Effects of activation slide"></a></td>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-10-interpretation.jpg" alt="Interpretation slide"></a></td>
    <td><a href="02-phase-2-imaging-analysis/presentation/"><img src="02-phase-2-imaging-analysis/presentation/previews/slide-11-take-home.jpg" alt="Take-home message slide"></a></td>
  </tr>
</table>

---

## 🚀 Roadmap

- [x] 🎓 Mentor confirmed and topic defined *(Phase 1)*
- [x] 🎤 First research deck drafted with preliminary findings *(Phase 2, 6/22)*
- [ ] 📓 Finish the image-analysis log
- [ ] 🖼️ Produce a clean, labeled figure set
- [ ] 🎞️ Complete the deck: add the ultrasound frames and the References slide
- [ ] 🔎 Confirm the source of every figure and cite it
- [ ] 🧊 Confirm with the mentor whether the Phase 1 3D numerical model is still expected

---

## 📖 Mini-glossary

<details>
<summary><b>Click to expand</b></summary>

| Term | Meaning |
|---|---|
| **PFM** | Pelvic floor muscle |
| **SUI** | Stress urinary incontinence: leakage during effort such as coughing |
| **ARJ** | Anorectal junction |
| **The Knack** | A voluntary pelvic floor contraction timed just before and during a cough |
| **Valsalva** | Bearing down against a closed airway, which raises abdominal pressure |
| **Cranial / caudal** | Toward the head / toward the feet |
| **Kinematic imaging** | Imaging that captures *movement* (displacement, velocity, acceleration) rather than a still picture |

</details>

---

## 🔒 Data & ethics

> [!WARNING]
> The imaging data belong to **Dr. Constantinou**. This repository contains processed clips, visualizations and reading material: no subject records, names or identifiers. Two journal PDFs and a 3D anatomy animation are **third-party material**. **No license is granted**; all rights are reserved. Please obtain the mentor's permission before reusing or redistributing any media. Details: [`docs/data-and-privacy.md`](docs/data-and-privacy.md).

## 🙏 Acknowledgements

Sincere thanks to **Dr. Christos E. Constantinou** for the mentorship, the data, and the guidance on clinically meaningful measurement and interpretation.

<div align="center">

<sub>Made with 💜 for curious minds · Preliminary work, not peer reviewed</sub>

</div>
