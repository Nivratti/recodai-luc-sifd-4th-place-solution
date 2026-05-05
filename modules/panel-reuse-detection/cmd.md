
## generating synthetic dataset pairs for  full duplicate, full overlap crop, and no match
python tools/generate_pairs.py   --config configs/gen_pairs.yaml   --input "resources/images/histology/85.jpg"   --out "./out/pairs_v3"

python tools/generate_pairs.py   --config configs/gen_pairs.yaml   --input "resources/images/histology/"   --out "./out/pairs_v4"