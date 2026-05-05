dataset:
  aug_root: out/aug_pos               # output root created by make_cbir_aug_positives.py
  manifest: manifest.jsonl            # relative to aug_root (or absolute path)

eval:
  topk_list: [1, 5, 10]
  store_topk: 10                      # store topK predictions in results.jsonl (for later analysis/viz)
  make_fail_viz: true
  fail_viz_k: 6
  fail_viz_thumb: 224
  max_fail_viz: 200                   # cap how many failure grids to save per model
  resume: true

defaults:
  device: cuda:0
  batch_size: 64
  fp16: true
  resize_mode: letterpad
  grayscale: false
  seed: 0
  deterministic: true

outputs:
  out_dir: out/cbir_aug_eval

models:
  - name: timm_resnet50
    backend: timm
    model_name: resnet50

  - name: timm_vgg16
    backend: timm
    model_name: vgg16

  - name: timm_dinov2_vits14
    backend: timm
    model_name: dinov2_vits14

  - name: sscd_disc_mixup
    backend: sscd
    sscd_torchscript_path: resources/models/sscd_disc_mixup.torchscript.pt
    img_size: 288
