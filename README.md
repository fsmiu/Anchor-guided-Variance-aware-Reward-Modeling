# Anchor-guided-Variance-aware-Reward-Modeling


Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python data/prepare_multipref.py
python data/prepare_helpsteer2.py
python data/prepare_helpsteer3.py
python data/prepare_personalllm.py
python data/prepare_rewardbench.py
python data/prepare_ppe_p.py
python data/prepare_ultrafeedback.py
```

Generate anchor score:

```bash
python scripts/generate_skywork_rewards.py
```

Reward modeling

```bash
bash scripts/run_anchor_multipref_split.sh
bash scripts/run_anchor_helpsteer2_split.sh
bash scripts/run_anchor_helpsteer3_split.sh
bash scripts/run_anchor_personalllm_split.sh
```

PPO:

```bash
bash scripts/run_ppo_ood_decouple.sh
```

Best of N:

```bash
bash scripts/run_bon_anchor_compare.sh
```