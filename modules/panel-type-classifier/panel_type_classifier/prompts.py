from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# ---------------------------
# Prompt pack tuned for paper panel crops (BiomedCLIP)
# ---------------------------

TEMPLATES = [
    "a photo of {}",
    "an image of {}",
    "a cropped scientific figure panel of {}",
    "a research paper figure panel showing {}",
    "a biomedical figure panel showing {}",
    "this figure panel shows {}",
    "a low-resolution paper panel showing {}",
    "a published figure panel containing {}",
]

# Stage A: imaging vs non-imaging
GROUP_PROMPTS = {
    "imaging": [
        # microscopy / micrographs
        "a microscopy image",
        "a micrograph of cells",
        "a fluorescence microscopy image",
        "a confocal microscopy image",
        "a brightfield microscopy image",
        "a phase contrast microscopy image",
        "an electron microscopy image",
        "a micrograph with a scale bar",
        "a merged multi-channel microscopy image",

        # histology/pathology slides
        "a histology tissue section image",
        "an H&E stained tissue section",
        "an immunohistochemistry (IHC) stained tissue image",
        "a pathology slide image",
        "a digital pathology whole slide image tile",

        # gels / blots
        "a western blot image",
        "a gel electrophoresis image",
        "an SDS-PAGE gel image",
        "a blot with lanes and bands",

        # radiology / scans
        "a chest x-ray image",
        "a CT scan image",
        "an MRI scan image",
        "an ultrasound image",
        "a PET scan image",

        # clinical photos / procedures
        "a clinical photograph",
        "a gross pathology specimen photo",
        "a dermatology clinical photo",
        "an endoscopy image",

        # ophthalmology
        "a fundus retina photograph",
        "an OCT retinal scan image",

        # plates/assays
        "a petri dish photo",
        "a colony assay plate image",
        "a multi-well plate photo",
    ],

    "non_imaging": [
        # plots/charts
        "a bar chart",
        "a line chart",
        "a scatter plot",
        "a box plot",
        "a violin plot",
        "a dot plot",
        "a volcano plot",
        "a ROC curve",
        "a Kaplan-Meier survival curve",
        "a PCA plot",
        "a UMAP plot",
        "a t-SNE plot",

        # tables/heatmaps
        "a table of numbers",
        "a heatmap plot",
        "a matrix heatmap with labels",

        # flow cytometry (usually treated as non-imaging)
        "a flow cytometry dot plot",
        "a flow cytometry histogram",
        "a FACS gating plot",

        # schematics/diagrams
        "a schematic diagram",
        "a pathway diagram",
        "a biological network diagram",
        "a flowchart",
        "a cartoon illustration",

        # molecular renderings / structures (non-imaging)
        "a protein ribbon diagram",
        "a molecular structure rendering",
        "a chemical structure diagram",

        # genomics/bioinfo visuals
        "a genome browser track plot",
        "a phylogenetic tree diagram",
        "a sequence alignment figure",

        # text panels / UI
        "a text-only panel",
        "a panel of labels and annotations",
        "a screenshot of software output",
    ],
}

# Stage B: imaging subtype (only if group_pred == imaging)
SUBTYPE_PROMPTS = {
    "blot_gel": [
        "a western blot membrane with lanes and bands",
        "a blot panel with protein bands",
        "a gel electrophoresis image with lanes",
        "an SDS-PAGE gel with a ladder",
        "a grayscale gel or blot panel",
        "a blot panel showing band intensity differences",
    ],

    "microscopy": [
        "a microscopy image of cells with a scale bar",
        "a fluorescence microscopy image with colored channels",
        "a confocal microscopy micrograph",
        "a brightfield microscopy micrograph",
        "a phase contrast microscopy image",
        "a merged multi-channel microscopy image",
        "an electron microscopy micrograph (TEM or SEM)",
        "a microscopy image showing nuclei stained with DAPI",
    ],

    "histology_pathology": [
        "an H&E stained histology tissue section",
        "a histopathology slide image of tissue",
        "an immunohistochemistry (IHC) stained tissue section",
        "a pathology biopsy slide image",
        "a digital pathology whole-slide scan tile",
        "a tissue section showing glands and stroma",
    ],

    "radiology": [
        "a chest x-ray radiograph",
        "a CT scan slice",
        "an MRI scan slice",
        "an ultrasound scan image",
        "a PET scan image",
        "a radiology medical imaging scan with grayscale anatomy",
    ],

    "clinical_photo": [
        "a clinical photograph of a patient condition",
        "a dermatology clinical photo of skin",
        "a gross pathology specimen photograph",
        "a surgical specimen photo",
        "a medical photograph taken with a camera",
    ],

    "ophthalmology": [
        "a fundus retina photograph",
        "an OCT retinal scan",
        "a fluorescein angiography retina image",
        "an ophthalmology clinical image of the eye",
    ],

    "endoscopy": [
        "an endoscopy image from inside the body",
        "a colonoscopy frame",
        "a bronchoscopy frame",
        "a laparoscopy image",
        "an internal organ endoscopic view",
    ],

    "assay_plate": [
        "a petri dish colony assay photo",
        "a bacterial colony plate image",
        "a spot assay plate image",
        "a multi-well plate photo",
        "an agar plate experiment photo",
    ],

    "other_imaging": [
        "a biomedical photograph that is not microscopy or radiology",
        "a device-captured biomedical image of unknown type",
        "a hard-to-categorize imaging panel",
        "a mixed imaging panel",
    ],
}



@dataclass(frozen=True)
class PromptPack:
    templates: List[str]
    group_prompts: Dict[str, List[str]]
    subtype_prompts: Dict[str, List[str]]


DEFAULT_PROMPT_PACK = PromptPack(
    templates=TEMPLATES,
    group_prompts=GROUP_PROMPTS,
    subtype_prompts=SUBTYPE_PROMPTS,
)