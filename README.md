# Schutz vor KI-Stilimitation: Glaze, Nightshade und Mist im Vergleich

Wie gut schützen Werkzeuge wie **Glaze**, **Nightshade** und **Mist** Bilder davor, dass ein Diffusionsmodell ihren Stil per Fine-Tuning übernimmt? Dieses Repository enthält das Experiment meiner Hausarbeit im Modul *Smart Graphics / Generative AI in Visual Computing* (M.Sc. Medieninformatik, Hochschule Emden/Leer).

Das Experiment beantwortet drei Fragen:

1. **Sichtbarkeit:** Wie stark verändern die Werkzeuge das Bild?
2. **Robustheit:** Übersteht der Schutz typische Web-Transformationen wie JPEG-Kompression, Skalierung oder Rauschen?
3. **Schutzwirkung:** Lernt ein LoRA auf geschützten Bildern den Stil schlechter als auf ungeschützten?

## Methode

| Baustein | Umsetzung |
|---|---|
| Datensatz | 15 Gemälde von Vincent van Gogh (gemeinfrei), zentriert auf 512×512 zugeschnitten |
| Schutz | Glaze und Nightshade (Desktop-Tools der University of Chicago), **Mist als eigene Re-Implementierung** in PyTorch (AdvDM-Verlust + Texturverlust, PGD mit L∞-Budget 8/255, 100 Schritte) nach Liang et al. (2023) |
| Robustheit | JPEG q=75/50/25, Skalierung 0,5×, Gauß-Rauschen σ=0,02 |
| Metriken | PSNR, SSIM, LPIPS (AlexNet), CLIP-Bildähnlichkeit (ViT-B/32) |
| Stilimitation | Je Variante ein DreamBooth-LoRA auf Stable Diffusion 1.5 (UNet-Attention, PEFT, Rank 4), danach Generierung mit identischen Prompts und Seeds |

**Stack:** Python, PyTorch, Hugging Face `diffusers` / `transformers` / `peft`, `lpips`, scikit-image, pandas, matplotlib. Ausgeführt auf NVIDIA-GPUs (Google Colab T4 und lokal).

## Ergebnisse

### Sichtbarkeit der Perturbation (Mittelwert über 15 Bilder)

| Werkzeug | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Glaze | 40,77 dB | 0,977 | 0,020 |
| Nightshade | 36,24 dB | 0,941 | 0,075 |
| Mist | 31,79 dB | 0,859 | 0,174 |

Glaze verändert die Bilder am wenigsten, Mist am stärksten.

### Robustheit: LPIPS(T(Original), T(geschützt))

| Transformation | Glaze | Nightshade | Mist |
|---|---|---|---|
| ohne (Referenz) | 0,020 | 0,075 | 0,174 |
| JPEG q=75 | 0,034 | 0,101 | 0,175 |
| JPEG q=50 | 0,041 | 0,093 | 0,147 |
| JPEG q=25 | 0,044 | 0,089 | 0,130 |
| Skalierung 0,5× | 0,014 | 0,039 | 0,129 |
| Rauschen σ=0,02 | 0,015 | 0,059 | 0,147 |

Herunterskalieren schwächt die Perturbation am deutlichsten ab, bei Nightshade etwa um die Hälfte.

### Stilimitation per LoRA

| Trainingsdaten | CLIP max ↑ | CLIP mittel ↑ | LPIPS min ↓ |
|---|---|---|---|
| Original (ungeschützt) | 0,726 | 0,616 | 0,763 |
| Glaze | 0,734 | 0,626 | 0,749 |
| Nightshade | 0,725 | 0,615 | 0,773 |
| Mist | 0,706 | 0,603 | 0,777 |

Die Unterschiede zwischen den Varianten sind gering. Nur Mist senkt die Ähnlichkeit der Imitationen zu den Originalen messbar.

**Einschränkung:** Die LoRAs wurden aus Rechenzeitgründen nur mit 50 Trainingsschritten trainiert. Die Ergebnisse zur Stilimitation sind deshalb als Tendenz zu verstehen, nicht als belastbarer Nachweis.

## Repository-Struktur

```
├── experiment_glaze_nightshade_mist.ipynb   # komplettes Experiment
├── data/
│   ├── original/       # 15 Ausgangsbilder
│   ├── glaze/
│   ├── nightshade/
│   └── mist/
└── results/            # Metriken (CSV + LaTeX) und Bildraster
```

## Ausführen

1. Notebook in Google Colab öffnen und als Laufzeit eine GPU wählen (T4 reicht).
2. Alle Zellen ausführen. Die Bilddaten werden aus diesem Repository geladen.

Glaze und Nightshade sind Closed-Source-Anwendungen und laufen außerhalb des Notebooks. Die geschützten Bilder liegen deshalb fertig in `data/`.

## Quellen

- Shan et al. (2023): *Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models*
- Shan et al. (2024): *Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models*
- Liang et al. (2023): *Mist: Towards Improved Adversarial Examples for Diffusion Models*
- Bildquellen: gemeinfreie Reproduktionen der Werke von Vincent van Gogh
